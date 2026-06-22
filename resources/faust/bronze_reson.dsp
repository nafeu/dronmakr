declare name "Bronze Reson";
declare description "Resonant FM tone with a bronze-like ring.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
ratio = hslider("ratio", 2.4, 1, 6, 0.01);
index = hslider("index", 5, 0, 12, 0.01);
res = hslider("res", 0.55, 0.1, 0.95, 0.01);

envelope = gain * en.adsr(0.002, 0.2, 0.35, 1.2, gate);
mod = os.osc(freq * ratio) * index * freq;
tone = os.osc(freq + mod);
process = tone : fi.resonlp(2, freq * 3.5, res) * envelope <: _, _;
