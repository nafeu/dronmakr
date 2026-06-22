declare name "Church Bell";
declare description "Large church bell strike with long tail.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
mod = os.osc(freq * 2.3) * freq * 4.5;
envelope = gain * en.adsr(0.002, 0.15, 0.25, 2.2, gate);
process = os.osc(freq + mod) * envelope <: _, _;
