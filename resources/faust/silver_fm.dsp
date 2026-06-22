declare name "Silver FM";
declare description "Silvery FM tone with a bright index.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
mod = os.osc(freq * 4.4) * freq * 6.5;
envelope = gain * en.adsr(0.002, 0.18, 0.28, 1.4, gate);
process = os.osc(freq + mod) * envelope <: _, _;
