declare name "Drawbar Organ";
declare description "Classic additive organ tone with drawbar harmonics.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");

envelope = gain * en.adsr(0.01, 0.08, 0.92, 0.25, gate);
voice = os.osc(freq)
      + 0.55 * os.osc(freq * 2)
      + 0.35 * os.osc(freq * 3)
      + 0.2 * os.osc(freq * 4);
process = voice * 0.45 * envelope <: _, _;
