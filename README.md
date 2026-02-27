Pipeline : 
# Etape 1 : Sauvegarde du fichier uploadé dans un fichier temporaire
# Etape 2 : Conversion et nettoyage de l'audio (ffmpeg)
# Etape 3 : Diarization (pyannote)
# Etape 4 : Merge les segments de parole si ils se suivent et qu'ils ont le même speaker
# Etape 5 : Pour chaque segment de parole identifié, on extrait le segment audio correspondant et on le transcrit avec whisper

Conteneurisation : 

docker build -f docker/Dockerfile -t transcribe-ai-multi-voix .

