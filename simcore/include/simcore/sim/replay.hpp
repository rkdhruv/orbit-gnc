#pragma once

#include <cstdint>
#include <span>

#include "simcore/sim/simulation.hpp"

namespace simcore {

// Re-run a session from its initial state and input log (Simulation::input_log) up to to_tick.
// Each entry's command is submitted when the fresh sim reaches the entry's tick, so it takes effect
// at that tick as it did originally. Because the state after tick n depends only on the initial
// state and the command in effect at each tick, the result is bit-identical to the original
// session's snapshot at to_tick (same machine and build). Entries at or after to_tick haven't
// taken effect yet and are ignored. Throws std::invalid_argument for to_tick < 0, entries not in
// strictly increasing tick order or at a negative tick, or an invalid command.
Snapshot replay(const SimConfig& cfg, const VehicleState& initial,
                std::span<const LoggedCommand> log, std::int64_t to_tick);

}  // namespace simcore
