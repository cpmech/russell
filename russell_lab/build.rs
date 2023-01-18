fn main() {
    cc::Build::new().file("c_code/math_functions.c").compile("c_code");
}
