#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include "jinja/caps.h"

#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char ** argv) {
    if (argc < 2) { std::cerr << "usage: caps_probe <template>\n"; return 2; }
    std::ifstream f(argv[1]);
    std::stringstream ss; ss << f.rdbuf();
    std::string src = ss.str();
    try {
        jinja::lexer lx; auto lexed = lx.tokenize(src);
        auto prog  = jinja::parse_from_tokens(lexed);
        auto c     = jinja::caps_get(prog);
        std::cout << c.to_string() << "\n";
        std::cout << "supports_object_arguments=" << (c.supports_object_arguments ? "true" : "false") << "\n";
        return 0;
    } catch (const std::exception & e) {
        std::cout << "EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}
