declare name "Mallet FM";
declare description "Percussive two-operator FM mallet.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
ratio = hslider("ratio", 3.2, 1, 8, 0.01);
index = hslider("index", 4.5, 0, 12, 0.01);

envelope = gain * en.ar(0.001, 0.35, gate);
mod = os.osc(freq * ratio) * index * freq;
process = os.osc(freq + mod) * envelope <: _, _;
