import re

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Find functions taking both a and b
    funcs_ab = ['t2_odyad_t2_slice', 't2_odyad_t2_update_slice', 't2_udyad_t2_slice', 'qsd_fn_slice']
    
    for func in funcs_ab:
        # dim == 4
        p4 = rf"(pub\(crate\) fn {func}\(.*?dim: usize\) {{\n(?:.*?)\n\s*if dim == 4 {{\n)"
        content = re.sub(p4, r"\1        let a = &a[..4];\n        let b = &b[..4];\n", content, count=1)
        
        # dim == 6
        p6 = rf"(pub\(crate\) fn {func}\(.*?dim: usize\) {{.*?\n\s*}} else if dim == 6 {{\n)"
        content = re.sub(p6, r"\1        let a = &a[..6];\n        let b = &b[..6];\n", content, flags=re.DOTALL, count=1)
        
        # dim == 9 (else)
        p9 = rf"(pub\(crate\) fn {func}\(.*?dim: usize\) {{.*?\n\s*}} else {{\n)"
        content = re.sub(p9, r"\1        let a = &a[..9];\n        let b = &b[..9];\n", content, flags=re.DOTALL, count=1)

    with open(filepath, 'w') as f:
        f.write(content)

process_file('/home/dorival/01-Code/rust/russell/russell_tensor/src/operations_mix2.rs')

# tensor2.rs functions taking a single slice or mut slice
def process_tensor2(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # transpose_slice
    content = content.replace(
        "pub(crate) fn transpose_slice(&self, at: &mut [f64]) {\n        if self.dim == 4 {\n",
        "pub(crate) fn transpose_slice(&self, at: &mut [f64]) {\n        if self.dim == 4 {\n            let at = &mut at[..4];\n"
    )
    content = content.replace(
        "        } else if self.dim == 6 {\n            at[0]",
        "        } else if self.dim == 6 {\n            let at = &mut at[..6];\n            at[0]"
    )
    content = content.replace(
        "        } else {\n            at[0]",
        "        } else {\n            let at = &mut at[..9];\n            at[0]"
    )

    # deviator_slice
    content = content.replace(
        "pub(crate) fn deviator_slice(&self, dev: &mut [f64]) {\n        let m = self.trace() / 3.0;\n        if self.dim == 4 {\n",
        "pub(crate) fn deviator_slice(&self, dev: &mut [f64]) {\n        let m = self.trace() / 3.0;\n        if self.dim == 4 {\n            let dev = &mut dev[..4];\n"
    )
    content = content.replace(
        "        } else if self.dim == 6 {\n            dev[0]",
        "        } else if self.dim == 6 {\n            let dev = &mut dev[..6];\n            dev[0]"
    )
    content = content.replace(
        "        } else {\n            dev[0]",
        "        } else {\n            let dev = &mut dev[..9];\n            dev[0]"
    )

    with open(filepath, 'w') as f:
        f.write(content)

process_tensor2('/home/dorival/01-Code/rust/russell/russell_tensor/src/tensor2.rs')
