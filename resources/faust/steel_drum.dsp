declare name "Steel Drum";
declare description "Caribbean steel drum FM strike.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
mod = os.osc(freq * 2.7) * freq * 5.5;
envelope = gain * en.ar(0.001, 0.55, gate);
process = os.osc(freq + mod) * envelope <: _, _;
