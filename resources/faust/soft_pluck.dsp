declare name "Soft Pluck";
declare description "Short decaying sine pluck for gentle motion.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
decay = hslider("decay", 0.45, 0.08, 1.5, 0.01);

envelope = gain * en.ar(0.002, decay, gate);
process = os.osc(freq) * envelope <: _, _;
