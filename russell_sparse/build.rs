fn main() {
    cc::Build::new()
        .file("c_code/main.c")
        .compile("c_code_main");
}
