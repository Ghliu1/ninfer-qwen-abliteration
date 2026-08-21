#include "options.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejected(std::vector<std::string> arguments) {
    try {
        (void)parse(std::move(arguments));
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

} // namespace

int main() {
    int failures = 0;
    const ninfer::cli::Options defaults =
        parse({"ninfer", "model.ninfer", "--prompt", "hello"});
    failures += check(defaults.max_merged_vision_tokens == 32768,
                      "CLI default merged Vision limit changed");

    const ninfer::cli::Options configured =
        parse({"ninfer", "model.ninfer", "--prompt", "hello", "--vision",
               "--max-merged-vision-tokens", "4096"});
    failures += check(configured.enable_vision && configured.max_merged_vision_tokens == 4096,
                      "CLI merged Vision limit was not preserved");

    failures += check(
        rejected({"ninfer", "model.ninfer", "--prompt", "hello",
                  "--max-merged-vision-tokens", "4096"}),
        "CLI accepted a merged Vision limit without --vision");
    failures +=
        check(rejected({"ninfer", "model.ninfer", "--prompt", "hello", "--vision",
                        "--max-merged-vision-tokens", "0"}),
              "CLI accepted a zero merged Vision limit");
    failures +=
        check(rejected({"ninfer", "model.ninfer", "--prompt", "hello", "--vision",
                        "--max-merged-vision-tokens", "32769"}),
              "CLI accepted a merged Vision limit above the frontend maximum");
    failures +=
        check(rejected({"ninfer", "model.ninfer", "--prompt", "hello", "--vision",
                        "--max-merged-vision-tokens", "bad"}),
              "CLI accepted a malformed merged Vision limit");
    failures += check(
        ninfer::cli::usage_text("ninfer").find("--max-merged-vision-tokens") !=
            std::string::npos,
        "CLI help omits the merged Vision limit");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
