{{/*
Expand the name of the chart.
*/}}
{{- define "sportsagentic-inference.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "sportsagentic-inference.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "sportsagentic-inference.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "sportsagentic-inference.labels" -}}
helm.sh/chart: {{ include "sportsagentic-inference.chart" . }}
{{ include "sportsagentic-inference.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "sportsagentic-inference.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sportsagentic-inference.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "sportsagentic-inference.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "sportsagentic-inference.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the Secret name holding HF_TOKEN
*/}}
{{- define "sportsagentic-inference.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- include "sportsagentic-inference.fullname" . }}
{{- end }}
{{- end }}

{{/*
Return the PersistentVolumeClaim name for model caching
*/}}
{{- define "sportsagentic-inference.pvcName" -}}
{{- if .Values.persistence.existingClaim }}
{{- .Values.persistence.existingClaim }}
{{- else }}
{{- printf "%s-models" (include "sportsagentic-inference.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Return the image tag as a string. A number from a values file or --reuse-values
is a float64 here, and toString would turn 8542126 into 8.542126e+06.
*/}}
{{- define "sportsagentic-inference.imageTag" -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- if kindIs "float64" $tag -}}
{{- if ne (floor $tag) $tag -}}
{{- fail (printf "image.tag %v is not a whole number, so YAML has already changed it. Quote it, or pass --set-string image.tag=<tag>." $tag) -}}
{{- end -}}
{{- printf "%.0f" $tag -}}
{{- else -}}
{{- $tag -}}
{{- end -}}
{{- end }}
