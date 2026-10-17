//! American English is the standard for this workspace.
//!
//! The list below is the set of British spellings that had crept into comments,
//! documentation and identifiers -- 139 occurrences across 26 files when the rule
//! was introduced. Keeping it as a test rather than a note in `AGENTS.md` means a
//! regression fails the suite instead of being caught by review.
//!
//! Scope: `.rs`, `.md` and `.py` files under the workspace root, excluding build
//! output and `.git`. Prose in `docs/` and the crates' READMEs counts: the wording
//! is read by users, and a codebase that mixes `behavior` and `behaviour` looks
//! like it has two authors.
//!
//! Deliberately *not* in the list:
//!
//! * `analyses` -- the plural of `analysis`, which is American English too
//!   ("finite element analyses"); only the verb `analyses` would be British, and
//!   the code has none.
//! * `advise`, `revise`, `exercise`, `surprise`, `precise`, `otherwise`, `wise`,
//!   `raise`, `devise`, `noise` -- `-ise` endings that are correct in American
//!   English.
//! * `columnwise`, `rowwise`, `componentwise`, `pairwise` -- not British forms.
//!
//! Adding an entry: put the British spelling here, in lowercase; the check is
//! case-insensitive and respects word boundaries, so `Normalise` and
//! `normalise` are both caught while `normalised_columns` (an identifier) is
//! caught too -- identifiers count.

use std::fs;
use std::path::Path;

/// British spellings, lowercase. The American form is not needed by the check
/// (it only looks for these), but is given in the comment for the next reader.
const BRITISH: &[&str] = &[
    // -our -> -or
    "behaviour", // behavior
    "behavioural",
    "behaviours",
    "colour",   // color
    "coloured", // colored
    "colours",
    "favour", // favor
    "favourite",
    "honour",    // honor
    "humour",    // humor
    "labour",    // labor
    "neighbour", // neighbor
    "neighbouring",
    "neighbours",
    "rumour", // rumor
    "vapour", // vapor
    // -ise -> -ize (geometry and numerics)
    "bidiagonalisation", // bidiagonalization
    "bidiagonalise",
    "bidiagonalises",
    "canonicalise", // canonicalize
    "characterise", // characterize
    "characterised",
    "characterisation",
    "customise",   // customize
    "diagonalise", // diagonalize
    "diagonalising",
    "discretise", // discretize
    "discretisation",
    "equalisation", // equalization
    "equalise",
    "factorisation", // factorization
    "factorisations",
    "factorise",
    "factorises",
    "generalise", // generalize
    "generalised",
    "initialisation", // initialization
    "initialise",
    "initialised",
    "initialises",
    "linearise",   // linearize
    "materialise", // materialize
    "materialising",
    "maximise", // maximize
    "maximises",
    "minimisation", // minimization
    "minimise",     // minimize
    "minimised",
    "modelling",     // modeling
    "monomorphise",  // monomorphize
    "normalisation", // normalization
    "normalise",     // normalize
    "normalised",
    "normalises",
    "optimisation", // optimization
    "optimise",     // optimize
    "optimised",
    "optimiser",
    "optimising",
    "organise", // organize
    "organised",
    "organisation",  // organization
    "orthogonalise", // orthogonalize
    "parameterise",  // parameterize
    "parameterisation",
    "prioritise",    // prioritize
    "randomise",     // randomize
    "rasterisation", // rasterization
    "rasterise",     // rasterize
    "rasterised",
    "rasteriser",
    "rasterises",
    "realise", // realize
    "realised",
    "recognise", // recognize
    "recognised",
    "serialise",  // serialize
    "specialise", // specialize
    "specialised",
    "stabilise",       // stabilize
    "standardisation", // standardization
    "standardise",     // standardize
    "standardised",
    "standardises",
    "summarise", // summarize
    "summarised",
    "symmetrise",   // symmetrize
    "tokenise",     // tokenize
    "unrecognised", // unrecognized
    "utilise",      // utilize
    "utilisation",  // utilization
    "vectorise",    // vectorize
    "visualise",    // visualize
    // -yse -> -yze
    "analyse", // analyze
    "analysed",
    "analysing", // analyzing
    // -re -> -er
    "calibre", // caliber
    "centre",  // center
    "centred", // centered
    "centres",
    "fibre", // fiber
    "litre", // liter
    "metre", // meter
    "metres",
    "sombre",  // somber
    "spectre", // specter
    "theatre", // theater
    // -ce -> -se
    "defence",  // defense
    "licence",  // license
    "offence",  // offense
    "pretence", // pretense
    "practise", // practice
    // doubled l -> single l
    "cancelled", // canceled
    "cancelling",
    "fuelled",  // fueled
    "labelled", // labeled
    "labelling",
    "levelled",   // leveled
    "modelled",   // modeled
    "signalling", // signaling
    "travelled",  // traveled
    // -ogue -> -og
    "analogue",  // analog
    "catalogue", // catalog
    "dialogue",  // dialog
    // misc
    "ageing",    // aging
    "aluminium", // aluminum
    "amongst",   // among
    "artefact",  // artifact
    "cheque",    // check
    "draught",   // draft
    "enrol",     // enroll
    "fulfil",    // fulfill
    "grey",      // gray
    "judgement", // judgment
    "kerb",      // curb
    "learnt",    // learned
    "maths",     // math
    "mould",     // mold
    "plough",    // plow
    "programme", // program
    "sceptical", // skeptical
    "spelt",     // spelled
    "storey",    // story
    "sulphur",   // sulfur
    "towards",   // toward
    "tyre",      // tire
    "whilst",    // while
];

fn is_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// True when `line` contains `word` with non-word characters (or the ends of the
/// line) on both sides -- an identifier counts as a word, so
/// `normalised_columns` is a hit.
fn contains_word(line_lower: &str, word: &str) -> bool {
    let mut from = 0;
    while let Some(rel) = line_lower[from..].find(word) {
        let start = from + rel;
        let end = start + word.len();
        let before_ok = line_lower[..start].chars().last().is_none_or(|c| !is_word_char(c));
        let after_ok = line_lower[end..].chars().next().is_none_or(|c| !is_word_char(c));
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
    }
    false
}

fn scan(dir: &Path, offenders: &mut Vec<String>) {
    for entry in fs::read_dir(dir).expect("read_dir") {
        let path = entry.expect("entry").path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if name == "target" || name == ".git" || name == "node_modules" {
            continue;
        }
        if path.is_dir() {
            scan(&path, offenders);
            continue;
        }
        let is_source = matches!(path.extension().and_then(|s| s.to_str()), Some("rs" | "md" | "py"));
        if !is_source {
            continue;
        }
        // This file is the one place the British spellings are written down.
        if name == "american_english.rs" {
            continue;
        }
        let Ok(text) = fs::read_to_string(&path) else { continue };
        for (i, line) in text.lines().enumerate() {
            let lower = line.to_lowercase();
            for word in BRITISH {
                if contains_word(&lower, word) {
                    offenders.push(format!("{}:{}: {}", path.display(), i + 1, line.trim()));
                    break;
                }
            }
        }
    }
}

#[test]
fn the_workspace_uses_american_english() {
    // The workspace root is one level above this crate.
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().expect("workspace root");
    let mut offenders = Vec::new();
    scan(root, &mut offenders);
    assert!(
        offenders.is_empty(),
        "{} British spelling(s) found -- this workspace uses American English:\n{}",
        offenders.len(),
        offenders.join("\n")
    );
}
