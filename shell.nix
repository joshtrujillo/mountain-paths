{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python312
    pkgs.python312Packages.numpy
    pkgs.python312Packages.folium
    pkgs.python312Packages.gpxpy
    pkgs.python312Packages.pip
    pkgs.python312Packages.matplotlib
    pkgs.python312Packages.requests
    pkgs.python312Packages.webcolors
    pkgs.python312Packages.sparqlwrapper
    pkgs.python312Packages.python-socketio
    pkgs.python312Packages.rasterio
    
    pkgs.ffmpeg
  ];

  #shellHook = ''
  #  if [ -d .venv ]; then
  #    source .venv/bin/activate
  #  else
  #    echo "Virtual environment .venv not found."
  #  fi
  #  '';
}
