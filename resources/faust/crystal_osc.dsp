declare name "Crystal Osc";
declare description "Light FM shimmer on a sine core.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.04, 0.2, 0.82, 0.45, gate);
mod = os.osc(freq * 2.01) * freq * 0.15;
process = os.osc(freq + mod) * envelope <: _, _;
