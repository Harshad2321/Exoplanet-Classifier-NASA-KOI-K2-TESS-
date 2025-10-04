
export interface ExoplanetData {
  orbitalPeriod: number;
  planetRadius: number;
  equilibriumTemperature: number;
  insolationFlux: number;
  distanceToStarRadius: number;
  stellarRadius: number;
  stellarMass: number;
  stellarTemperature: number;
  stellarAge: number;
  rightAscension: number;
  declination: number;
  dispositionScore: number;
}

export enum Classification {
  CONFIRMED = 'CONFIRMED',
  CANDIDATE = 'CANDIDATE',
  FALSE_POSITIVE = 'FALSE_POSITIVE'
}

export interface ClassificationResponse {
  classification: Classification;
  rationale: string;
}

export interface InputFieldConfig {
    id: string;
    label: string;
    unit: string;
    tooltip: string;
    defaultValue: number;
    step: number;
    precision: number;
}
