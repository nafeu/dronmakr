declare name "Harpsichord Tick";
declare description "Bright, percussive square tick with fast decay.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
decay = hslider("decay", 0.18, 0.05, 0.6, 0.01);

envelope = gain * en.ar(0.001, decay, gate);
process = os.square(freq) * envelope * 0.65 <: _, _;
