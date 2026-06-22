declare name "Buzz Stack";
declare description "Aggressive saw and square blend.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.03, 0.15, 0.8, 0.3, gate);
process = (os.sawtooth(freq) + os.square(freq)) * 0.45 * envelope <: _, _;
