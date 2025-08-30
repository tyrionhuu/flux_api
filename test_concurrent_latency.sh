 for i in {1..10}; do
    # NGINX API
    curl -X POST http://localhost:8080/generate \
      -H "Content-Type: application/json" \
      -d '{
        "prompt": "test image '$i'",
        "width": 512,
        "height": 512
      }' & 

    # # SINGLE API
    # curl -X POST http://localhost:23333/generate \
    #   -H "Content-Type: application/json" \
    #   -d '{
    #     "prompt": "test image '$i'",
    #     "width": 512,
    #     "height": 512
    #   }' & 
  done

wait