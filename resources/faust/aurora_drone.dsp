declare name "Aurora Drone";
declare description "Northern lights drone with slow shimmer.";
freq = nentry("freq", 440, 20, 20000, 1);
gain = nentry("gain", 0.5, 0, 1, 0.01);
gate = button("gate");
envelope = gain * en.adsr(0.5, 0.8, 0.64, 1.9, gate);
shimmer = 0.65 + 0.35 * os.lf_triangle(0.14);
voice = os.triangle(freq) + 0.3*os.triangle(freq*1.01);
process = voice * envelope * shimmer * 0.55 <: _, _;
