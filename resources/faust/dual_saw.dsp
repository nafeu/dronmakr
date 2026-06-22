declare name "Dual Saw";
declare description "Detuned saw pair for wide analog-style leads.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
spread = hslider("spread", 12, 0, 40, 0.1);

envelope = gain * en.adsr(0.04, 0.18, 0.82, 0.35, gate);
voice = (os.sawtooth(freq * (1 + spread / 1200)) + os.sawtooth(freq * (1 - spread / 1200))) * 0.5;
process = voice * envelope <: _, _;
