fn main() {
    cc::Build::new()
        .file("c_code/erf.c")
        .compile("c_code_erf");
}
