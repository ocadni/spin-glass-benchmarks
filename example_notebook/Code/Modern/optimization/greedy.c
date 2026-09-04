#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#ifdef USE_DOUBLE
typedef double real_t;
#else
typedef float real_t;
#endif

#define DOWNHILL_EPS ((real_t)1.0e-6)
#define CACHELINE 64

typedef struct {
    int u, v;
    real_t w;
} Edge;

typedef struct {
    int n;
    size_t m;
    Edge *edges;          /* each undirected edge once */
    real_t *site_field;   /* external/local field h_i from the instance */
    int has_fields;       /* whether an N-line field block was present */
    size_t *off;          /* CSR offsets, n+1 */
    int *nbr;             /* CSR neighbours, 2m */
    real_t *weight;       /* CSR weights, 2m */
} Graph;

typedef enum { MODE_RANDOM = 0, MODE_RELUCTANT = 1 } Mode;

typedef struct {
    const char *instance;
    long pop_size;
    long sweeps;
    Mode mode;
    int num_spins;
    uint64_t seed;
    int seed_given;
    int zero_fields;
} Options;

typedef struct {
    uint64_t s0, s1;
} RNG;

typedef struct {
    int8_t *spin;
    real_t *effective_field; /* h_i + sum_j J_ij s_j (or pairwise part only with --zero_fields true) */

    /* random mode: dense O(1) removable set of downhill spins */
    int *moves;
    int *pos;
    int move_count;

    /* reluctant mode: indexed max heap ordered by delta E (<0, closest to zero) */
    int *heap;
    int *hpos;
    uint64_t *tie;
    int heap_size;
} Workspace;

typedef struct {
    double final_energy;
    long flips;
} ReplicaResult;

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(EXIT_FAILURE);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n ? n : 1, s ? s : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xaligned(size_t n) {
    void *p = NULL;
    size_t bytes = n ? n : 1;
    if (posix_memalign(&p, CACHELINE, bytes) != 0 || !p) die("aligned allocation failed");
    return p;
}

static inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static inline uint64_t rotl64(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

/* xoroshiro128+ */
static inline uint64_t rng_next(RNG *r) {
    uint64_t s0 = r->s0;
    uint64_t s1 = r->s1;
    uint64_t out = s0 + s1;
    s1 ^= s0;
    r->s0 = rotl64(s0, 55) ^ s1 ^ (s1 << 14);
    r->s1 = rotl64(s1, 36);
    return out;
}

static inline void rng_seed(RNG *r, uint64_t seed) {
    uint64_t x = seed;
    r->s0 = splitmix64(&x);
    r->s1 = splitmix64(&x);
    if ((r->s0 | r->s1) == 0) r->s1 = 1;
}

static inline uint64_t rng_bounded(RNG *r, uint64_t bound) {
    if (bound <= 1) return 0;
#if defined(__SIZEOF_INT128__)
    return (uint64_t)(((__uint128_t)rng_next(r) * (__uint128_t)bound) >> 64);
#else
    /* fallback: tiny modulo bias, irrelevant for this heuristic */
    return rng_next(r) % bound;
#endif
}

static inline double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s INSTANCE --num-spins N [options]\n"
        "\n"
        "Fast CPU C implementation of the greedy spin-glass search.\n"
        "\n"
        "Options:\n"
        "  --pop-size N           replicas (default 1000)\n"
        "  --sweeps N             max greedy sweeps (default 5)\n"
        "  --mode random|reluctant (default random)\n"
        "  --num-spins N          REQUIRED; authoritative number of spins\n"
        "  --zero_fields BOOL     true: ignore h_i; false: use h_i (default false)\n"
        "  --seed N               RNG seed\n"
        "  -h, --help             show this help\n"
        "\n"
        "Input format:\n"
        "  first non-empty line: # ... metadata/comment ...\n"
        "  optional field block: N lines `i h_i` with zero-based i\n"
        "  then pairwise couplings: lines `i j J_ij`\n"
        "If the first data line has 3 columns, the field block is absent and h_i=0.\n",
        prog);
}

static long parse_long(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "error: invalid %s: %s\n", name, s);
        exit(EXIT_FAILURE);
    }
    return v;
}

static uint64_t parse_u64(const char *s, const char *name) {
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) {
        fprintf(stderr, "error: invalid %s: %s\n", name, s);
        exit(EXIT_FAILURE);
    }
    return (uint64_t)v;
}

static int parse_bool(const char *s, const char *name) {
    if (!strcmp(s, "true") || !strcmp(s, "1")) return 1;
    if (!strcmp(s, "false") || !strcmp(s, "0")) return 0;
    fprintf(stderr, "error: invalid %s: %s (expected true or false)\n", name, s);
    exit(EXIT_FAILURE);
}

static Options parse_args(int argc, char **argv) {
    Options o;
    memset(&o, 0, sizeof(o));
    o.pop_size = 1000;
    o.sweeps = 5;
    o.mode = MODE_RANDOM;

    if (argc < 2) {
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    if (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help")) {
        usage(argv[0]);
        exit(EXIT_SUCCESS);
    }
    o.instance = argv[1];

    for (int i = 2; i < argc; ++i) {
        const char *a = argv[i];
#define NEED_VALUE() do { if (++i >= argc) { fprintf(stderr, "error: %s requires a value\n", a); exit(EXIT_FAILURE); } } while (0)
        if (!strcmp(a, "--pop-size")) {
            NEED_VALUE(); o.pop_size = parse_long(argv[i], "pop-size");
        } else if (!strcmp(a, "--sweeps")) {
            NEED_VALUE(); o.sweeps = parse_long(argv[i], "sweeps");
        } else if (!strcmp(a, "--mode")) {
            NEED_VALUE();
            if (!strcmp(argv[i], "random")) o.mode = MODE_RANDOM;
            else if (!strcmp(argv[i], "reluctant")) o.mode = MODE_RELUCTANT;
            else die("--mode must be random or reluctant");
        } else if (!strcmp(a, "--num-spins")) {
            NEED_VALUE();
            long n = parse_long(argv[i], "num-spins");
            if (n <= 0 || n > INT32_MAX) die("--num-spins must be in [1, INT32_MAX]");
            o.num_spins = (int)n;
        } else if (!strcmp(a, "--zero_fields")) {
            NEED_VALUE(); o.zero_fields = parse_bool(argv[i], "zero_fields");
        } else if (!strcmp(a, "--seed")) {
            NEED_VALUE(); o.seed = parse_u64(argv[i], "seed"); o.seed_given = 1;
        } else if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            usage(argv[0]); exit(EXIT_SUCCESS);
        } else {
            fprintf(stderr, "error: unknown option: %s\n", a);
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
#undef NEED_VALUE
    }

    if (o.pop_size <= 0) die("--pop-size must be > 0");
    if (o.sweeps < 0) die("--sweeps must be >= 0");
    if (o.num_spins <= 0) die("--num-spins N is mandatory and must be > 0");
    return o;
}

static int parse_numeric_columns(const char *line, double vals[3]) {
    char buf[4096];
    size_t len = strlen(line);
    if (len >= sizeof(buf)) die("input line too long");
    memcpy(buf, line, len + 1);

    /* Allow trailing inline comments. */
    char *hash = strchr(buf, '#');
    if (hash) *hash = '\0';

    char *p = buf;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
    if (!*p) return 0;

    int count = 0;
    while (*p) {
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
        if (!*p) break;
        if (count == 3) return 4; /* too many columns */
        char *end = NULL;
        errno = 0;
        vals[count] = strtod(p, &end);
        if (errno || end == p) return -1;
        ++count;
        p = end;
    }
    return count;
}

static int exact_index(double x, int n, size_t line_no, const char *what) {
    if (!isfinite(x) || x < 0.0 || x >= (double)n || floor(x) != x) {
        fprintf(stderr, "error: invalid %s index %.17g on line %zu; expected integer in [0,%d)\n",
                what, x, line_no, n);
        exit(EXIT_FAILURE);
    }
    return (int)x;
}

static Graph load_graph(const Options *o) {
    FILE *f = fopen(o->instance, "r");
    if (!f) {
        fprintf(stderr, "error: cannot open %s: %s\n", o->instance, strerror(errno));
        exit(EXIT_FAILURE);
    }

    const int n = o->num_spins;
    real_t *site_field = (real_t *)xcalloc((size_t)n, sizeof(*site_field));
    unsigned char *field_seen = (unsigned char *)xcalloc((size_t)n, 1);

    size_t cap = 4096, m = 0;
    Edge *edges = (Edge *)xmalloc(cap * sizeof(*edges));
    char line[4096];
    size_t line_no = 0;
    int saw_first_nonempty = 0;
    int data_mode = 0; /* 0 undecided, 2 field block, 3 coupling block */
    int fields_read = 0;
    int has_fields = 0;

    while (fgets(line, sizeof(line), f)) {
        ++line_no;
        char *p = line;
        while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') ++p;
        if (!*p) continue;

        if (!saw_first_nonempty) {
            saw_first_nonempty = 1;
            if (*p != '#') {
                fprintf(stderr, "error: first non-empty line must start with '#'; got line %zu: %s",
                        line_no, line);
                exit(EXIT_FAILURE);
            }
            continue; /* header contents are metadata only */
        }

        if (*p == '#') continue;

        double v[3] = {0.0, 0.0, 0.0};
        int cols = parse_numeric_columns(p, v);
        if (cols == 0) continue;
        if (cols < 0 || cols > 3) {
            fprintf(stderr, "error: malformed numeric data on line %zu: %s", line_no, line);
            exit(EXIT_FAILURE);
        }

        if (data_mode == 0) {
            if (cols == 2) {
                data_mode = 2;
                has_fields = 1;
            } else if (cols == 3) {
                data_mode = 3;
                has_fields = 0;
            } else {
                fprintf(stderr, "error: first data row on line %zu must be `i h_i` or `i j J_ij`\n",
                        line_no);
                exit(EXIT_FAILURE);
            }
        }

        if (data_mode == 2 && fields_read < n) {
            if (cols != 2) {
                fprintf(stderr,
                        "error: expected %d field rows `i h_i`; coupling-like row encountered after %d fields on line %zu\n",
                        n, fields_read, line_no);
                exit(EXIT_FAILURE);
            }
            int i = exact_index(v[0], n, line_no, "field");
            if (!isfinite(v[1])) {
                fprintf(stderr, "error: non-finite field on line %zu\n", line_no);
                exit(EXIT_FAILURE);
            }
            if (field_seen[i]) {
                fprintf(stderr, "error: duplicate field for spin %d on line %zu\n", i, line_no);
                exit(EXIT_FAILURE);
            }
            field_seen[i] = 1;
            site_field[i] = (real_t)v[1];
            ++fields_read;
            if (fields_read == n) data_mode = 3;
            continue;
        }

        if (data_mode == 3) {
            if (cols != 3) {
                fprintf(stderr, "error: expected coupling row `i j J_ij` on line %zu\n", line_no);
                exit(EXIT_FAILURE);
            }
            int u = exact_index(v[0], n, line_no, "coupling");
            int vv = exact_index(v[1], n, line_no, "coupling");
            if (u == vv) {
                fprintf(stderr, "error: self-edge (%d,%d) on line %zu is not allowed\n", u, vv, line_no);
                exit(EXIT_FAILURE);
            }
            if (!isfinite(v[2])) {
                fprintf(stderr, "error: non-finite coupling on line %zu\n", line_no);
                exit(EXIT_FAILURE);
            }
            if (m == cap) {
                cap *= 2;
                Edge *tmp = (Edge *)realloc(edges, cap * sizeof(*edges));
                if (!tmp) die("out of memory growing edge list");
                edges = tmp;
            }
            edges[m++] = (Edge){u, vv, (real_t)v[2]};
        }
    }
    fclose(f);

    if (!saw_first_nonempty) die("empty instance file");
    if (has_fields && fields_read != n) {
        fprintf(stderr, "error: field block has %d rows, but --num-spins=%d\n", fields_read, n);
        exit(EXIT_FAILURE);
    }
    if (has_fields) {
        for (int i = 0; i < n; ++i) {
            if (!field_seen[i]) {
                fprintf(stderr, "error: field block is missing spin index %d\n", i);
                exit(EXIT_FAILURE);
            }
        }
    }
    free(field_seen);

    if (m == 0) die("no pairwise couplings were parsed from the instance");

    size_t *deg = (size_t *)xcalloc((size_t)n, sizeof(*deg));
    for (size_t e = 0; e < m; ++e) {
        ++deg[edges[e].u];
        ++deg[edges[e].v];
    }

    size_t *off = (size_t *)xmalloc((size_t)(n + 1) * sizeof(*off));
    off[0] = 0;
    for (int i = 0; i < n; ++i) off[i + 1] = off[i] + deg[i];
    if (off[n] != 2 * m) die("internal CSR size mismatch");

    int *nbr = (int *)xaligned(2 * m * sizeof(*nbr));
    real_t *weight = (real_t *)xaligned(2 * m * sizeof(*weight));
    size_t *cursor = (size_t *)xmalloc((size_t)n * sizeof(*cursor));
    memcpy(cursor, off, (size_t)n * sizeof(*cursor));

    for (size_t e = 0; e < m; ++e) {
        int u = edges[e].u, v = edges[e].v;
        real_t w = edges[e].w;
        size_t a = cursor[u]++;
        size_t b = cursor[v]++;
        nbr[a] = v; weight[a] = w;
        nbr[b] = u; weight[b] = w;
    }
    free(cursor);
    free(deg);

    Graph g = {n, m, edges, site_field, has_fields, off, nbr, weight};
    return g;
}

static void free_graph(Graph *g) {
    free(g->edges);
    free(g->site_field);
    free(g->off);
    free(g->nbr);
    free(g->weight);
    memset(g, 0, sizeof(*g));
}

static Workspace workspace_create(int n, Mode mode) {
    Workspace w;
    memset(&w, 0, sizeof(w));
    w.spin = (int8_t *)xaligned((size_t)n * sizeof(*w.spin));
    w.effective_field = (real_t *)xaligned((size_t)n * sizeof(*w.effective_field));
    if (mode == MODE_RANDOM) {
        w.moves = (int *)xaligned((size_t)n * sizeof(*w.moves));
        w.pos = (int *)xaligned((size_t)n * sizeof(*w.pos));
    } else {
        w.heap = (int *)xaligned((size_t)n * sizeof(*w.heap));
        w.hpos = (int *)xaligned((size_t)n * sizeof(*w.hpos));
        w.tie = (uint64_t *)xaligned((size_t)n * sizeof(*w.tie));
    }
    return w;
}

static void workspace_free(Workspace *w) {
    free(w->spin); free(w->effective_field);
    free(w->moves); free(w->pos);
    free(w->heap); free(w->hpos); free(w->tie);
    memset(w, 0, sizeof(*w));
}

static inline real_t delta_e(const Workspace *w, int i) {
    return (real_t)2 * (real_t)w->spin[i] * w->effective_field[i];
}

static inline int is_downhill(const Workspace *w, int i) {
    return delta_e(w, i) < -DOWNHILL_EPS;
}

/* ---------- O(1) active-set operations for random mode ---------- */
static inline void set_add(Workspace *w, int i) {
    int p = w->move_count++;
    w->moves[p] = i;
    w->pos[i] = p;
}

static inline void set_remove(Workspace *w, int i) {
    int p = w->pos[i];
    if (p < 0) return;
    int lastp = --w->move_count;
    int last = w->moves[lastp];
    if (p != lastp) {
        w->moves[p] = last;
        w->pos[last] = p;
    }
    w->pos[i] = -1;
}

static inline void set_refresh(Workspace *w, int i) {
    int down = is_downhill(w, i);
    if (down) {
        if (w->pos[i] < 0) set_add(w, i);
    } else if (w->pos[i] >= 0) {
        set_remove(w, i);
    }
}

/* ---------- indexed heap for reluctant mode ---------- */
static inline int heap_better(const Workspace *w, int a, int b) {
    real_t da = delta_e(w, a), db = delta_e(w, b);
    if (da > db) return 1;  /* both negative: larger = closer to zero */
    if (da < db) return 0;
    return w->tie[a] > w->tie[b];
}

static inline void heap_swap(Workspace *w, int a, int b) {
    int ia = w->heap[a], ib = w->heap[b];
    w->heap[a] = ib; w->heap[b] = ia;
    w->hpos[ia] = b; w->hpos[ib] = a;
}

static inline void heap_sift_up(Workspace *w, int p) {
    while (p > 0) {
        int parent = (p - 1) >> 1;
        if (!heap_better(w, w->heap[p], w->heap[parent])) break;
        heap_swap(w, p, parent);
        p = parent;
    }
}

static inline void heap_sift_down(Workspace *w, int p) {
    for (;;) {
        int l = p * 2 + 1;
        if (l >= w->heap_size) break;
        int r = l + 1;
        int best = l;
        if (r < w->heap_size && heap_better(w, w->heap[r], w->heap[l])) best = r;
        if (!heap_better(w, w->heap[best], w->heap[p])) break;
        heap_swap(w, p, best);
        p = best;
    }
}

static inline void heap_insert(Workspace *w, int i) {
    int p = w->heap_size++;
    w->heap[p] = i;
    w->hpos[i] = p;
    heap_sift_up(w, p);
}

static inline void heap_remove(Workspace *w, int i) {
    int p = w->hpos[i];
    if (p < 0) return;
    int lastp = --w->heap_size;
    w->hpos[i] = -1;
    if (p == lastp) return;
    int x = w->heap[lastp];
    w->heap[p] = x;
    w->hpos[x] = p;
    if (p > 0 && heap_better(w, x, w->heap[(p - 1) >> 1])) heap_sift_up(w, p);
    else heap_sift_down(w, p);
}

static inline void heap_refresh(Workspace *w, int i, RNG *rng) {
    int down = is_downhill(w, i);
    int p = w->hpos[i];
    if (!down) {
        if (p >= 0) heap_remove(w, i);
        return;
    }
    w->tie[i] = rng_next(rng); /* robust random tie breaking */
    if (p < 0) {
        heap_insert(w, i);
    } else {
        p = w->hpos[i];
        if (p > 0 && heap_better(w, i, w->heap[(p - 1) >> 1])) heap_sift_up(w, p);
        else heap_sift_down(w, p);
    }
}

static double initialize_replica(const Graph *g, Workspace *w, Mode mode, RNG *rng, int zero_fields) {
    const int n = g->n;
    double energy = 0.0;
    for (int i = 0; i < n; ++i) {
        w->spin[i] = (rng_next(rng) & 1) ? (int8_t)1 : (int8_t)-1;
        w->effective_field[i] = zero_fields ? (real_t)0 : g->site_field[i];
        if (!zero_fields) {
            energy -= (double)g->site_field[i] * (double)w->spin[i];
        }
    }
    for (size_t e = 0; e < g->m; ++e) {
        int u = g->edges[e].u, v = g->edges[e].v;
        real_t ew = g->edges[e].w;
        int su = w->spin[u], sv = w->spin[v];
        w->effective_field[u] += ew * (real_t)sv;
        w->effective_field[v] += ew * (real_t)su;
        energy -= (double)ew * (double)su * (double)sv;
    }

    if (mode == MODE_RANDOM) {
        memset(w->pos, 0xff, (size_t)n * sizeof(*w->pos));
        w->move_count = 0;
        for (int i = 0; i < n; ++i) if (is_downhill(w, i)) set_add(w, i);
    } else {
        memset(w->hpos, 0xff, (size_t)n * sizeof(*w->hpos));
        w->heap_size = 0;
        for (int i = 0; i < n; ++i) {
            if (is_downhill(w, i)) {
                w->tie[i] = rng_next(rng);
                heap_insert(w, i);
            }
        }
    }
    return energy;
}

static inline void apply_flip_random(const Graph *g, Workspace *w, int k, double *energy) {
    const int old = w->spin[k];
    const real_t de = delta_e(w, k);
    *energy += (double)de;
    w->spin[k] = (int8_t)-old;

    /* Only neighbours' local fields change. */
    size_t begin = g->off[k], end = g->off[k + 1];
    for (size_t a = begin; a < end; ++a) {
        int j = g->nbr[a];
        w->effective_field[j] -= (real_t)2 * g->weight[a] * (real_t)old;
    }

    /* Only k and its neighbours can change downhill status. */
    set_refresh(w, k);
    for (size_t a = begin; a < end; ++a) set_refresh(w, g->nbr[a]);
}

static inline void apply_flip_reluctant(const Graph *g, Workspace *w, int k, RNG *rng, double *energy) {
    const int old = w->spin[k];
    const real_t de = delta_e(w, k);
    *energy += (double)de;
    w->spin[k] = (int8_t)-old;

    size_t begin = g->off[k], end = g->off[k + 1];
    for (size_t a = begin; a < end; ++a) {
        int j = g->nbr[a];
        w->effective_field[j] -= (real_t)2 * g->weight[a] * (real_t)old;
    }

    heap_refresh(w, k, rng);
    for (size_t a = begin; a < end; ++a) heap_refresh(w, g->nbr[a], rng);
}

static ReplicaResult run_replica(const Graph *g, Workspace *w, const Options *o,
                                 uint64_t replica_seed, long total_steps) {
    RNG rng;
    rng_seed(&rng, replica_seed);
    double energy = initialize_replica(g, w, o->mode, &rng, o->zero_fields);
    long flips = 0;

    while (flips < total_steps) {
        int k;
        if (o->mode == MODE_RANDOM) {
            if (w->move_count == 0) break;
            int p = (int)rng_bounded(&rng, (uint64_t)w->move_count);
            k = w->moves[p];
            apply_flip_random(g, w, k, &energy);
        } else {
            if (w->heap_size == 0) break;
            k = w->heap[0];
            apply_flip_reluctant(g, w, k, &rng, &energy);
        }
        ++flips;
    }

    ReplicaResult r = {energy, flips};
    return r;
}

int main(int argc, char **argv) {
    Options o = parse_args(argc, argv);
    if (!o.seed_given) {
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        o.seed = ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec ^ (uint64_t)(uintptr_t)&o;
    }

    Graph g = load_graph(&o);
    if (o.sweeps > 0 && o.sweeps > LONG_MAX / g.n) die("sweeps*N overflows long");
    long total_steps = o.sweeps * (long)g.n;
    ReplicaResult *res = (ReplicaResult *)xmalloc((size_t)o.pop_size * sizeof(*res));

    double t0 = now_seconds();

    Workspace w = workspace_create(g.n, o.mode);
    for (long p = 0; p < o.pop_size; ++p) {
        uint64_t x = o.seed ^ (UINT64_C(0xd1b54a32d192ed03) * (uint64_t)(p + 1));
        uint64_t rs = splitmix64(&x);
        res[p] = run_replica(&g, &w, &o, rs, total_steps);
    }
    workspace_free(&w);

    double elapsed = now_seconds() - t0;

    long max_flips = 0;
    double min_final = INFINITY;
    double sum_final = 0.0;
    for (long p = 0; p < o.pop_size; ++p) {
        if (res[p].flips > max_flips) max_flips = res[p].flips;
        double e = res[p].final_energy / (double)g.n;
        if (e < min_final) min_final = e;
        sum_final += e;
    }
    int early_stop = (max_flips < total_steps);
    double mean_final = sum_final / (double)o.pop_size;

    if (early_stop) {
        printf("[%s] All replicas reached a local minimum at step %ld / %ld.\n",
               o.mode == MODE_RANDOM ? "RANDOM" : "RELUCTANT", max_flips, total_steps);
    }

    printf("instance: %s\n", o.instance);
    printf("N: %d\n", g.n);
    printf("edges: %zu\n", g.m);
    printf("fields_in_file: %s\n", g.has_fields ? "yes" : "no");
    printf("zero_fields: %s\n", o.zero_fields ? "true" : "false");
    printf("mode: %s\n", o.mode == MODE_RANDOM ? "random" : "reluctant");
    printf("pop_size: %ld\n", o.pop_size);
    printf("sweeps: %ld\n", o.sweeps);
    printf("min_energy_per_spin: %.8f\n", min_final);
    printf("final_mean_energy_per_spin: %.8f\n", mean_final);
    printf("elapsed_seconds: %.6f\n", elapsed);
    printf("seed: %" PRIu64 "\n", o.seed);

    free(res);
    free_graph(&g);
    return 0;
}
