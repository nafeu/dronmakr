import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

export type AudioValidation = {
  ok: boolean;
  path: string;
  durationS: number;
  sampleRate: number;
  peak: number;
  rms: number;
};

function pythonBin(): string {
  const venvPython = path.join(process.cwd(), "venv/bin/python");
  if (fs.existsSync(venvPython)) {
    return venvPython;
  }
  return "python3";
}

export function validateWavFile(wavPath: string): AudioValidation {
  const script = path.join(process.cwd(), "scripts/e2e/validate_audio.py");
  const out = execFileSync(pythonBin(), [script, wavPath, "--json"], {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  return JSON.parse(out) as AudioValidation;
}
