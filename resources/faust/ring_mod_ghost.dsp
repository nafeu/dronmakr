declare name "Ring Mod Ghost";
declare description "Haunted ring-mod-like partial clash.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.01, 0.2, 0.35, 1.1, gate);
voice = os.osc(freq * 1.41) * os.osc(freq * 0.71);
process = voice * envelope * 0.8 <: _, _;
