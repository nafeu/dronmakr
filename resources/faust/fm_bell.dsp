declare name "FM Bell";
declare description "Bright bell partials from frequency modulation.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
ratio = hslider("ratio", 3.5, 1, 9, 0.01);
index = hslider("index", 6, 0, 14, 0.01);

envelope = gain * en.adsr(0.002, 0.08, 0.2, 1.4, gate);
mod = os.osc(freq * ratio) * index * freq;
process = os.osc(freq + mod) * envelope <: _, _;
