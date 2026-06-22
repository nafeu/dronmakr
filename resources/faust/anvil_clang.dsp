declare name "Anvil Clang";
declare description "Heavy anvil clang with dense partials.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.001, 0.12, 0.18, 1.0, gate);
voice = os.osc(freq) + os.osc(freq*2.83) + os.osc(freq*4.55) + os.osc(freq*6.12);
process = voice * 0.24 * envelope <: _, _;
