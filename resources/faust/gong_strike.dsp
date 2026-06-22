declare name "Gong Strike";
declare description "Large gong with a long metallic bloom.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
mod = os.osc(freq * 1.7) * freq * 3.2;
envelope = gain * en.adsr(0.002, 0.25, 0.2, 2.5, gate);
process = os.osc(freq + mod) * envelope * 0.85 <: _, _;
