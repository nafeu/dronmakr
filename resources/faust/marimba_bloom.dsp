declare name "Marimba Bloom";
declare description "Wooden marimba bloom with FM body.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
mod = os.osc(freq * 3.1) * freq * 1.6;
envelope = gain * en.ar(0.001, 0.55, gate);
process = os.osc(freq + mod) * envelope * 0.8 <: _, _;
