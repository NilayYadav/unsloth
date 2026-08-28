#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "ggml_abort at %s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}
