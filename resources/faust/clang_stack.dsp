declare name "Clang Stack";
declare description "Inharmonic partial stack for metallic clangs.";

freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");

envelope = gain * en.adsr(0.001, 0.12, 0.15, 1.0, gate);
voice = os.osc(freq * 1.0)
      + os.osc(freq * 2.71)
      + os.osc(freq * 4.16)
      + os.osc(freq * 5.43);
process = voice * 0.25 * envelope <: _, _;
