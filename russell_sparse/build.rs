fn main() {
    // println!("cargo:rustc-link-lib=dylib=openblas");
    // println!("cargo:rustc-link-lib=dylib=lapacke");
    cc::Build::new().file("c_code.c").compile("foo");
    /*
    cc::Build::new()
        .cpp(true)
        .file("wrapper.cpp")
        .cpp_link_stdlib("stdc++")
        .compile("libwrapper.a");
        */
}
