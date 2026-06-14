#[cfg(feature = "cudss")]
#[link(name = "interface_cudss", kind = "static")]
extern "C" {
    fn run_hello_world();
}

#[cfg(feature = "cudss")]
pub fn call_cuda() {
    unsafe {
        run_hello_world();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#[cfg(all(test, feature = "cudss"))]
mod tests {
    use super::call_cuda;

    #[test]
    fn hello_world() {
        println!("Hello World: Begin");
        call_cuda();
        println!("Hello World: End");
    }
}
