//! oxe-server: an OpenAI-compatible HTTP server shaped like llama-server, over the
//! [`engine::Engine`] trait (real engine: crates/model once M2 lands; [`engine::StubEngine`]
//! until then). See docs/lab-notes/SERVER.md.

pub mod api;
pub mod cache;
pub mod echo;
pub mod engine;
pub mod fair;
#[cfg(feature = "flash")]
pub mod flash_engine;
pub mod flash_switches;
pub mod generate;
pub mod http;
pub mod mtp_guard;
pub mod oxe_engine;
pub mod prefix_store;
pub mod sampling;
pub mod service;
pub mod startup_check;
pub mod state_store;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_echo;
#[cfg(test)]
mod tests_flash_prefix;
#[cfg(test)]
mod tests_flash_prod;
#[cfg(test)]
mod tests_hotfix;
#[cfg(test)]
mod tests_lan_share;
#[cfg(test)]
mod tests_long_append;
#[cfg(test)]
mod tests_mtp_sampled;
#[cfg(test)]
mod tests_oxe_state;
#[cfg(test)]
mod tests_state;
#[cfg(test)]
mod tests_state_clear;
#[cfg(test)]
mod tests_state_p2b;
#[cfg(test)]
mod tests_state_v013;
#[cfg(test)]
mod tests_store_depth;
#[cfg(test)]
mod tests_v013_int;
#[cfg(test)]
mod tests_v014_session;
#[cfg(test)]
mod tests_v020_fair;
#[cfg(test)]
mod tests_v020_hints;
#[cfg(test)]
mod tests_v020_legacy;
#[cfg(test)]
mod tests_v020_openai;
#[cfg(test)]
mod tests_v020_rewrite;
#[cfg(test)]
mod tests_v020_round1;
#[cfg(test)]
mod tests_v020_timings;
