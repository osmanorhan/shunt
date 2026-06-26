//! shunt-bench — offline scope-resolution smoke + pointer to the live suite.
//!
//! Usage:
//!   cargo run -p shunt-bench              # offline eval fixtures (no model)
//!
//! The real agentic feedback loop lives in `live.rs` and drives the unified
//! session core (the same path as `agent --once` / the TUI). It needs a model
//! server and is gated behind FRAME_TEST_ENDPOINT:
//!
//!   FRAME_TEST_ENDPOINT=http://127.0.0.1:8080 \
//!     cargo test -p shunt-bench live -- --nocapture --test-threads=1

use shunt_bench::eval;

fn main() {
    println!("Frame offline eval fixtures (scope resolution, scripted provider)\n");

    let fixtures = [
        eval::add_dependency_rust(),
        eval::add_dependency_npm(),
        eval::modify_existing_file(),
        eval::scaffold_new_file(),
    ];

    let mut passed = 0;
    for fixture in &fixtures {
        let result = eval::run_fixture(fixture);
        result.print();
        if result.passed {
            passed += 1;
        }
    }

    println!("\n{passed}/{} eval fixtures passed", fixtures.len());
    println!(
        "For the agentic feedback loop, run the live suite (see the module docs in src/live.rs)."
    );
}
