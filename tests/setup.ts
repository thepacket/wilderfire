// The JWildfire ports are loaded lazily in the app (main.ts awaits loadJwfVariations());
// tests get the complete registry up front.
import { loadJwfVariations } from '../src/core/variations';

await loadJwfVariations();
