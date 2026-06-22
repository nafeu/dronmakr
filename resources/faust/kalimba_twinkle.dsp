declare name "Kalimba Twinkle";
declare description "Bright kalimba-like FM twinkle.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
mod = os.osc(freq * 4.2) * freq * 2.2;
envelope = gain * en.ar(0.001, 0.42, gate);
process = os.osc(freq + mod) * envelope <: _, _;
