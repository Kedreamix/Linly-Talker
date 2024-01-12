# OpenSSL Certificate for HTTPS

When launching Gradio locally, you may encounter HTTPS-related issues like microphone access being blocked. To resolve this, you can use OpenSSL to configure HTTPS.

## Generate Certificate and Key

Use OpenSSL to generate a self-signed certificate and private key:

```
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 365 -nodes
```

This will generate `cert.pem` and `key.pem` in the current directory.

## Launch Gradio with HTTPS

Launch Gradio by specifying the certificate and key: 

```python
demo.launch(
  server_name="0.0.0.0",
  ssl_certfile="/path/to/cert.pem", 
  ssl_keyfile="/path/to/key.pem"
)
```

By providing the SSL certificate and key, Gradio will launch an HTTPS server instead of HTTP.

The benefits of HTTPS include:

- Enabling browser features like microphone access
- Avoiding mixed content issues
- Encrypted communication

In summary, generating a self-signed certificate with OpenSSL and providing it to Gradio enables launching a local HTTPS server, resolving limitations like microphone blocking. This improves the user experience when interacting with the model.

