#include "simcore/sim/replay.hpp"

#include "simcore/check.hpp"

namespace simcore {

Snapshot replay(const SimConfig& cfg, const VehicleState& initial,
                std::span<const LoggedCommand> log, std::int64_t to_tick) {
    SIMCORE_CHECK(to_tick >= 0, "to_tick must be >= 0");
    for (std::size_t i = 0; i < log.size(); ++i) {
        SIMCORE_CHECK(log[i].tick >= 0, "log entries must be at ticks >= 0");
        SIMCORE_CHECK(i == 0 || log[i].tick > log[i - 1].tick,
                      "log entries must be in strictly increasing tick order");
    }
    Simulation sim(cfg, initial);
    std::int64_t tick = 0;
    for (const LoggedCommand& e : log) {
        if (e.tick >= to_tick) {
            break;
        }
        sim.step(e.tick - tick);
        tick = e.tick;
        sim.submit(e.command);  // validated; takes effect at the start of tick e.tick
    }
    sim.step(to_tick - tick);
    return sim.snapshot();
}

}  // namespace simcore
