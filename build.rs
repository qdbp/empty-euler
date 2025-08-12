use std::{
    env, fs,
    io::Write,
    path::{Path, PathBuf},
};

fn main() {
    let root = Path::new("ps");
    println!("cargo:rerun-if-changed={}", root.display());

    let mut files = vec![];
    walk(root, &mut files);
    files.sort();

    let mut mods = String::new();
    let mut reg = String::from("pub type Solver = fn(&[String]) -> anyhow::Result<String>;\n");
    reg.push_str("pub static REGISTRY: &[(&str, Solver)] = &[\n");

    for rel in files {
        let stem = Path::new(&rel).file_stem().unwrap().to_str().unwrap();
        let absolute = fs::canonicalize(&rel).unwrap();
        let abs_path = absolute.to_str().unwrap();
        mods.push_str(&format!("#[path = \"{abs_path}\"] pub mod {stem};\n"));
        reg.push_str(&format!(
            "    (\"{stem}\", crate::infra::solutions::{stem}::solve as Solver),\n",
        ));
        println!("cargo:rerun-if-changed={rel}");
    }
    reg.push_str("];\n");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    write_if_changed(out.join("solutions_mod.rs"), mods.as_bytes());
    write_if_changed(out.join("registry.rs"), reg.as_bytes());
}

fn walk(dir: &Path, out: &mut Vec<String>) {
    for entry in fs::read_dir(dir).unwrap() {
        let p = entry.unwrap().path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.extension().and_then(|s| s.to_str()) == Some("rs") {
            let stem = p.file_stem().unwrap().to_str().unwrap();
            if stem.starts_with('p') && stem[1..].chars().all(|c| c.is_ascii_digit()) {
                out.push(p.to_string_lossy().into_owned());
            }
        }
    }
}
fn write_if_changed(path: PathBuf, bytes: &[u8]) {
    if fs::read(&path).ok().as_deref() != Some(bytes) {
        fs::File::create(&path).unwrap().write_all(bytes).unwrap();
    }
}
