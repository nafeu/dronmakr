declare name "Music Box";
declare description "Gentle music box FM tone.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
mod = os.osc(freq * 2.8) * freq * 1.2;
envelope = gain * en.adsr(0.002, 0.12, 0.35, 1.0, gate);
process = os.osc(freq + mod) * envelope * 0.75 <: _, _;
