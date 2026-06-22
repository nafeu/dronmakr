declare name "Crystal FM";
declare description "Glassy high-ratio FM with a sparkling decay.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
ratio = hslider("ratio", 5.8, 2, 12, 0.01);
index = hslider("index", 7, 0, 16, 0.01);

envelope = gain * en.adsr(0.003, 0.15, 0.25, 1.6, gate);
mod = os.osc(freq * ratio) * index * freq;
process = os.osc(freq + mod) * envelope * 0.8 <: _, _;
