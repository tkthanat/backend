# app/auth.py
import httpx
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from typing import Dict, Any

# --- (ID ของคุณ ถูกต้องแล้ว) ---
TENANT_ID = "0ecb7c82-1b84-4b36-adef-2081b5c1125b"
CLIENT_ID = "af39ad67-ec03-4cbd-88f3-762dd7a58dfe"

# (เราไม่จำเป็นต้องใช้ AUDIENCE_ID ที่นี่)
# AUDIENCE_ID = "api://af39ad67-ec03-4cbd-88f3-762dd7a58dfe"
# ---

AUTH_ISSUER = f"https://login.microsoftonline.com/{TENANT_ID}/v2.0"
JWKS_URL = f"https://login.microsoftonline.com/{TENANT_ID}/discovery/v2.0/keys"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

_jwks_cache: Dict[str, Any] = {}


async def get_jwks() -> Dict[str, Any]:
    global _jwks_cache
    if _jwks_cache:
        return _jwks_cache

    async with httpx.AsyncClient() as client:
        try:
            print(f"Fetching JWKS from: {JWKS_URL}")
            response = await client.get(JWKS_URL)
            response.raise_for_status()
            jwks = response.json()
            _jwks_cache = {key['kid']: key for key in jwks['keys']}
            print("Successfully fetched and cached JWKS.")
            return _jwks_cache
        except Exception as e:
            print(f"Failed to fetch JWKS: {e}")
            _jwks_cache = {}
            raise HTTPException(status_code=500, detail=f"Could not fetch auth keys from provider: {e}")


async def get_token_claims(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    try:
        unverified_header = jwt.get_unverified_header(token)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    kid = unverified_header.get("kid")
    if not kid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 'kid' (Key ID) not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    jwks = await get_jwks()

    rsa_key_data = jwks.get(kid)
    if not rsa_key_data:
        print(f"Public key '{kid}' not found in cache. Refetching...")
        global _jwks_cache
        _jwks_cache = {}
        jwks = await get_jwks()
        rsa_key_data = jwks.get(kid)
        if not rsa_key_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Public key not found for token after refetch",
                headers={"WWW-Authenticate": "Bearer"},
            )

    try:
        payload = jwt.decode(
            token,
            rsa_key_data,
            algorithms=["RS256"],
            # ✨ [นี่คือจุดแก้ไขที่สำคัญที่สุด] ✨
            # เราต้องตรวจสอบกับ CLIENT_ID (af39...)
            # ไม่ใช่ AUDIENCE_ID (api://af39...)
            audience=CLIENT_ID,
            issuer=AUTH_ISSUER
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.JWTClaimsError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid claims: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_name(claims: Dict[str, Any] = Depends(get_token_claims)) -> str:
    user_name = claims.get("name")
    if not user_name:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User name not found in token",
        )
    return user_name