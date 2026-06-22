declare name "Tick Tock";
declare description "Tiny percussive tick for rhythmic motion.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.ar(0.0005, 0.08, gate);
process = os.square(freq) * envelope * 0.45 <: _, _;
