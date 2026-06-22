declare name "Copper Bowl";
declare description "Singing copper bowl with slow decay.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
mod = os.osc(freq * 2.1) * freq * 2.8;
envelope = gain * en.adsr(0.003, 0.2, 0.3, 2.0, gate);
process = os.osc(freq + mod) : fi.lowpass(1, 6000) * envelope <: _, _;
