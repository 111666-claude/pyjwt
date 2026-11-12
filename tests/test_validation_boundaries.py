"""Security boundary tests for JWT validation.

These tests exercise the validation edges that matter for algorithm
confusion and anomalous-time-value attacks:

* the caller's algorithm allow-list always wins over the token header
* ``alg: none`` can never verify a signature
* symmetric and asymmetric keys are never cross-interpreted
* duplicated JSON members in header/payload are rejected
* exp/nbf/iat and leeway share one clock semantics
* error paths never echo key material back to the caller

All keys are generated in memory (nothing is read from disk) and every
time-dependent test runs against a fixed clock via the ``_now`` seam.
The suite only depends on the standard library plus ``cryptography`` and
is runnable directly with ``python -m unittest``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import unittest
from datetime import timedelta

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

import jwt
from jwt.api_jwt import PyJWT
from jwt.exceptions import (
    DecodeError,
    ExpiredSignatureError,
    ImmatureSignatureError,
    InvalidAlgorithmError,
    InvalidIssuedAtError,
    InvalidKeyError,
    InvalidSignatureError,
    InvalidTokenError,
    PyJWTError,
)
from jwt.utils import base64url_encode

try:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    HAS_ED25519 = True
except ImportError:  # pragma: no cover
    HAS_ED25519 = False


# A fixed point in time: 2025-01-01T00:00:00Z
FIXED_NOW = 1735689600.0


class FixedClockPyJWT(PyJWT):
    """PyJWT instance whose validation clock is pinned to FIXED_NOW."""

    @staticmethod
    def _now() -> float:
        return FIXED_NOW


def b64url(raw: bytes) -> str:
    return base64url_encode(raw).decode()


def make_token(header: dict | bytes, payload: dict | bytes, signature: bytes) -> str:
    if isinstance(header, dict):
        header = json.dumps(header, separators=(",", ":")).encode()
    if isinstance(payload, dict):
        payload = json.dumps(payload, separators=(",", ":")).encode()
    return ".".join([b64url(header), b64url(payload), b64url(signature)])


def hmac_sign(signing_input: bytes, secret: bytes, alg="sha256") -> bytes:
    return hmac.new(secret, signing_input, getattr(hashlib, alg)).digest()


class InMemoryKeys:
    """Generate every key used by the suite in memory."""

    @classmethod
    def setUpKeys(cls) -> None:
        cls.hmac_secret = b"in-memory-hmac-secret-at-least-32-bytes!!"
        cls.hmac_secret_other = b"another-in-memory-hmac-secret-32b!!"

        cls.rsa_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        cls.rsa_public = cls.rsa_private.public_key()
        cls.rsa_public_pem = cls.rsa_public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        cls.rsa_private_pem = cls.rsa_private.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        cls.ec_private = ec.generate_private_key(ec.SECP256R1())
        cls.ec_public = cls.ec_private.public_key()
        cls.ec_public_pem = cls.ec_public.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

        if HAS_ED25519:
            cls.ed_private = Ed25519PrivateKey.generate()
            cls.ed_public = cls.ed_private.public_key()
            cls.ed_public_pem = cls.ed_public.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )


class AlgorithmWhitelistTests(InMemoryKeys, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setUpKeys()
        cls.verifier = FixedClockPyJWT()

    def test_token_alg_not_in_allowlist_is_rejected(self) -> None:
        token = jwt.encode({"a": 1}, self.hmac_secret, algorithm="HS384")
        with self.assertRaises(InvalidAlgorithmError):
            jwt.decode(token, self.hmac_secret, algorithms=["HS256"])

    def test_non_string_alg_header_is_rejected(self) -> None:
        for alg_value in (123, True, None, ["HS256"], {"alg": "HS256"}):
            with self.subTest(alg=alg_value):
                token = make_token({"alg": alg_value, "typ": "JWT"}, {"a": 1}, b"")
                with self.assertRaises(InvalidTokenError):
                    jwt.decode(token, self.hmac_secret, algorithms=["HS256", "none"])

    def test_empty_alg_header_is_rejected(self) -> None:
        token = make_token({"alg": "", "typ": "JWT"}, {"a": 1}, b"")
        with self.assertRaises(InvalidAlgorithmError):
            jwt.decode(token, self.hmac_secret, algorithms=["HS256", "none"])

    def test_allowlisted_alg_still_requires_valid_signature(self) -> None:
        token = jwt.encode({"a": 1}, self.hmac_secret, algorithm="HS256")
        with self.assertRaises(InvalidSignatureError):
            jwt.decode(token, self.hmac_secret_other, algorithms=["HS256"])

    def test_algorithms_argument_is_mandatory_for_verification(self) -> None:
        token = jwt.encode({"a": 1}, self.hmac_secret, algorithm="HS256")
        with self.assertRaises(DecodeError):
            jwt.decode(token, self.hmac_secret)  # no algorithms=

    def test_algorithms_empty_list_rejects_every_token(self) -> None:
        token = jwt.encode({"a": 1}, self.hmac_secret, algorithm="HS256")
        # An empty allow-list can never match a token; the library treats
        # the missing/empty argument as a caller configuration error.
        with self.assertRaises(InvalidTokenError):
            jwt.decode(token, self.hmac_secret, algorithms=[])

    def test_rs_token_rejected_when_only_hmac_allowed(self) -> None:
        token = jwt.encode({"a": 1}, self.rsa_private, algorithm="RS256")
        with self.assertRaises(InvalidAlgorithmError):
            jwt.decode(token, self.rsa_public_pem, algorithms=["HS256"])

    def test_hs_token_rejected_when_only_rsa_allowed(self) -> None:
        token = jwt.encode({"a": 1}, self.hmac_secret, algorithm="HS256")
        with self.assertRaises(InvalidAlgorithmError):
            jwt.decode(token, self.hmac_secret, algorithms=["RS256"])


class NoneAlgorithmTests(InMemoryKeys, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setUpKeys()

    def test_none_token_fails_even_when_none_is_allowlisted(self) -> None:
        token = jwt.encode({"a": 1}, key=None, algorithm="none")
        with self.assertRaises(DecodeError):
            jwt.decode(token, None, algorithms=["none"])

    def test_none_token_rejected_by_default_allowlist(self) -> None:
        token = jwt.encode({"a": 1}, key=None, algorithm="none")
        with self.assertRaises(InvalidAlgorithmError):
            jwt.decode(token, None, algorithms=["HS256"])

    def test_none_with_nonempty_signature_segment_still_fails(self) -> None:
        token = make_token(
            {"alg": "none", "typ": "JWT"},
            {"a": 1},
            b"garbage-signature",
        )
        with self.assertRaises(InvalidSignatureError):
            jwt.decode(token, None, algorithms=["none"])

    def test_hs256_token_presented_with_none_header_is_rejected(self) -> None:
        # Strip the signature but keep a none header; the empty signature
        # never verifies and the caller's HMAC allow-list rejects none too.
        token = make_token({"alg": "none"}, {"a": 1}, b"")
        with self.assertRaises(InvalidAlgorithmError):
            jwt.decode(token, self.hmac_secret, algorithms=["HS256"])

    def test_none_algorithm_rejects_nonempty_key(self) -> None:
        from jwt.algorithms import NoneAlgorithm

        with self.assertRaises(InvalidKeyError):
            NoneAlgorithm().prepare_key("not-empty")


class KeyTypeSeparationTests(InMemoryKeys, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setUpKeys()

    def test_rsa_public_key_cannot_be_used_as_hmac_secret(self) -> None:
        # The classic algorithm-confusion forgery: sign with HMAC over the
        # PEM public key and claim alg=HS256.
        header = b64url(b'{"alg":"HS256","typ":"JWT"}').encode()
        payload = b64url(b'{"a":1}').encode()
        signing_input = header + b"." + payload
        signature = hmac_sign(signing_input, self.rsa_public_pem)
        forged = b".".join([header, payload, base64url_encode(signature)]).decode()
        with self.assertRaises(InvalidKeyError):
            jwt.decode(forged, self.rsa_public_pem, algorithms=["HS256"])

    def test_ec_public_key_cannot_be_used_as_hmac_secret(self) -> None:
        header = b64url(b'{"alg":"HS256","typ":"JWT"}').encode()
        payload = b64url(b'{"a":1}').encode()
        signing_input = header + b"." + payload
        signature = hmac_sign(signing_input, self.ec_public_pem)
        forged = b".".join([header, payload, base64url_encode(signature)]).decode()
        with self.assertRaises(InvalidKeyError):
            jwt.decode(forged, self.ec_public_pem, algorithms=["HS256"])

    def test_hmac_secret_cannot_be_loaded_as_rsa_key(self) -> None:
        from jwt.algorithms import RSAAlgorithm
        from cryptography.hazmat.primitives import hashes

        with self.assertRaises(InvalidKeyError):
            RSAAlgorithm(hashes.SHA256).prepare_key(self.hmac_secret)

    def test_rsa_key_object_rejected_by_hmac_prepare_key(self) -> None:
        from jwt.algorithms import HMACAlgorithm

        for key in (self.rsa_public, self.rsa_private, self.ec_public):
            with self.subTest(key=type(key).__name__):
                with self.assertRaises(TypeError):
                    HMACAlgorithm(hashlib.sha256).prepare_key(key)

    def test_ec_curve_mismatch_is_rejected(self) -> None:
        # Same advertised algorithm but an incompatible curve: the key's
        # curve must be bound to the algorithm, not inferred from the token.
        p384_private = ec.generate_private_key(ec.SECP384R1())
        p384_public_pem = p384_private.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        token = jwt.encode({"a": 1}, p384_private, algorithm="ES384")
        with self.assertRaises(PyJWTError):
            jwt.decode(token, p384_public_pem, algorithms=["ES256"])

    def test_jwk_like_string_rejected_as_hmac_secret(self) -> None:
        jwk_like = json.dumps({"kty": "oct", "k": b64url(self.hmac_secret)})
        with self.assertRaises(InvalidKeyError):
            jwt.encode({"a": 1}, jwk_like, algorithm="HS256")

    @unittest.skipUnless(HAS_ED25519, "ed25519 requires a recent cryptography")
    def test_ed25519_public_key_cannot_be_used_as_hmac_secret(self) -> None:
        header = b64url(b'{"alg":"HS256","typ":"JWT"}').encode()
        payload = b64url(b'{"a":1}').encode()
        signing_input = header + b"." + payload
        signature = hmac_sign(signing_input, self.ed_public_pem)
        forged = b".".join([header, payload, base64url_encode(signature)]).decode()
        with self.assertRaises(InvalidKeyError):
            jwt.decode(forged, self.ed_public_pem, algorithms=["HS256"])


class DuplicateHeaderTests(InMemoryKeys, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setUpKeys()

    def test_duplicate_alg_none_then_hs256_is_rejected(self) -> None:
        header = b'{"alg":"none","alg":"HS256","typ":"JWT"}'
        h = base64url_encode(header)
        p = base64url_encode(b'{"a":1}')
        sig = hmac_sign(h + b"." + p, self.hmac_secret)
        token = b".".join([h, p, base64url_encode(sig)]).decode()
        with self.assertRaises(DecodeError):
            jwt.decode(token, self.hmac_secret, algorithms=["HS256"])

    def test_duplicate_alg_hs256_then_none_is_rejected(self) -> None:
        header = b'{"alg":"HS256","alg":"none","typ":"JWT"}'
        h = base64url_encode(header)
        p = base64url_encode(b'{"a":1}')
        token = b".".join([h, p, base64url_encode(b"")]).decode()
        with self.assertRaises(DecodeError):
            jwt.decode(token, None, algorithms=["none"])

    def test_duplicate_other_header_member_is_rejected(self) -> None:
        header = b'{"alg":"HS256","typ":"JWT","kid":"a","kid":"b"}'
        h = base64url_encode(header)
        p = base64url_encode(b'{"a":1}')
        sig = hmac_sign(h + b"." + p, self.hmac_secret)
        token = b".".join([h, p, base64url_encode(sig)]).decode()
        with self.assertRaises(DecodeError):
            jwt.decode(token, self.hmac_secret, algorithms=["HS256"])

    def test_duplicate_payload_member_is_rejected(self) -> None:
        h = base64url_encode(b'{"alg":"HS256","typ":"JWT"}')
        p = base64url_encode(b'{"exp":1,"exp":' + str(FIXED_NOW + 100).encode() + b"}")
        sig = hmac_sign(h + b"." + p, self.hmac_secret)
        token = b".".join([h, p, base64url_encode(sig)]).decode()
        with self.assertRaises(DecodeError):
            FixedClockPyJWT().decode(token, self.hmac_secret, algorithms=["HS256"])

    def test_get_unverified_header_rejects_duplicates_too(self) -> None:
        h = base64url_encode(b'{"alg":"HS256","alg":"none"}')
        token = b".".join([h, base64url_encode(b"{}"), base64url_encode(b"")]).decode()
        with self.assertRaises(DecodeError):
            jwt.get_unverified_header(token)

    def test_nested_duplicate_keys_are_rejected(self) -> None:
        h = base64url_encode(b'{"alg":"HS256","typ":"JWT"}')
        p = base64url_encode(b'{"obj":{"k":1,"k":2}}')
        sig = hmac_sign(h + b"." + p, self.hmac_secret)
        token = b".".join([h, p, base64url_encode(sig)]).decode()
        with self.assertRaises(DecodeError):
            jwt.decode(token, self.hmac_secret, algorithms=["HS256"])


class TimeClaimTests(InMemoryKeys, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setUpKeys()
        cls.verifier = FixedClockPyJWT()

    def decode(self, payload: dict, **kwargs):
        token = jwt.encode(payload, self.hmac_secret, algorithm="HS256")
        return self.verifier.decode(
            token, self.hmac_secret, algorithms=["HS256"], **kwargs
        )

    def test_exp_in_future_is_accepted(self) -> None:
        self.decode({"exp": FIXED_NOW + 60})

    def test_exp_in_past_is_rejected(self) -> None:
        with self.assertRaises(ExpiredSignatureError):
            self.decode({"exp": FIXED_NOW - 1})

    def test_leeway_extends_expiration_window(self) -> None:
        # Expired 30s ago, 60s leeway -> still acceptable.
        self.decode({"exp": FIXED_NOW - 30}, leeway=60)
        self.decode({"exp": FIXED_NOW - 30}, leeway=timedelta(seconds=60))

    def test_leeway_does_not_save_a_long_expired_token(self) -> None:
        with self.assertRaises(ExpiredSignatureError):
            self.decode({"exp": FIXED_NOW - 120}, leeway=60)

    def test_negative_leeway_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.decode({"exp": FIXED_NOW + 100}, leeway=-1)
        with self.assertRaises(ValueError):
            self.decode({"exp": FIXED_NOW + 100}, leeway=timedelta(seconds=-1))

    def test_non_finite_leeway_is_rejected(self) -> None:
        # NaN would make every time comparison False, silently disabling
        # expiration; infinity would make every token valid forever.
        expired = {"exp": FIXED_NOW - 100}
        for bad_leeway in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(leeway=bad_leeway):
                with self.assertRaises(ValueError):
                    self.decode(expired, leeway=bad_leeway)

    def test_nbf_in_future_is_rejected(self) -> None:
        with self.assertRaises(ImmatureSignatureError):
            self.decode({"nbf": FIXED_NOW + 60})

    def test_nbf_in_past_is_accepted(self) -> None:
        self.decode({"nbf": FIXED_NOW - 60})

    def test_leeway_relaxes_nbf(self) -> None:
        # Becomes valid 30s in the future; 60s leeway covers it.
        self.decode({"nbf": FIXED_NOW + 30}, leeway=60)

    def test_iat_in_future_is_rejected(self) -> None:
        with self.assertRaises(ImmatureSignatureError):
            self.decode({"iat": FIXED_NOW + 60})

    def test_iat_in_past_is_accepted(self) -> None:
        self.decode({"iat": FIXED_NOW - 60})

    def test_all_three_claims_share_one_clock_reading(self) -> None:
        # A token valid at exactly one instant: iat/nbf in the past,
        # exp in the future relative to the single pinned clock.
        decoded = self.decode(
            {
                "iat": FIXED_NOW - 10,
                "nbf": FIXED_NOW - 10,
                "exp": FIXED_NOW + 10,
            }
        )
        self.assertIn("exp", decoded)

    def test_boolean_time_claims_are_rejected(self) -> None:
        for claim in ("exp", "nbf", "iat"):
            for value in (True, False):
                with self.subTest(claim=claim, value=value):
                    expected = InvalidIssuedAtError if claim == "iat" else DecodeError
                    with self.assertRaises(expected):
                        self.decode({claim: value})

    def test_non_numeric_time_claims_are_rejected(self) -> None:
        for claim in ("exp", "nbf", "iat"):
            for value in ("not-a-number", [1], {"x": 1}, None):
                with self.subTest(claim=claim, value=value):
                    expected = InvalidIssuedAtError if claim == "iat" else DecodeError
                    with self.assertRaises(expected):
                        self.decode({claim: value})

    def test_numeric_string_claims_still_coerce(self) -> None:
        # Backwards compatibility: numeric strings have historically been
        # accepted; the value must still be evaluated numerically.
        with self.assertRaises(ExpiredSignatureError):
            self.decode({"exp": str(int(FIXED_NOW) - 100)})
        self.decode({"exp": str(int(FIXED_NOW) + 100)})

    def test_extremely_large_timestamp_is_handled(self) -> None:
        huge = 10**30
        self.decode({"exp": huge})
        # A huge negative timestamp is simply far in the past -> expired,
        # without any OverflowError escaping the library.
        with self.assertRaises(ExpiredSignatureError):
            self.decode({"exp": -(10**30)})

    def test_float_timestamp_preserves_fractional_seconds(self) -> None:
        # 0.25s in the future at the pinned clock must remain valid even
        # though integer truncation would make it equal to now.
        self.decode({"exp": FIXED_NOW + 0.25})
        with self.assertRaises(ExpiredSignatureError):
            self.decode({"exp": FIXED_NOW - 0.25})

    def test_inf_and_nan_timestamps_are_rejected(self) -> None:
        raw_inf = b'{"exp": Infinity}'
        raw_nan = b'{"exp": NaN}'
        for raw in (raw_inf, raw_nan):
            with self.subTest(raw=raw):
                h = base64url_encode(b'{"alg":"HS256","typ":"JWT"}')
                p = base64url_encode(raw)
                sig = hmac_sign(h + b"." + p, self.hmac_secret)
                token = b".".join([h, p, base64url_encode(sig)]).decode()
                with self.assertRaises(DecodeError):
                    self.verifier.decode(token, self.hmac_secret, algorithms=["HS256"])


class NoKeyLeakageTests(InMemoryKeys, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setUpKeys()
        cls.marker = b"LEAKMARKER-super-secret-value-0123456789"

    def assertNoMarker(self, error: BaseException) -> None:
        self.assertNotIn(self.marker, str(error).encode())

    def test_rsa_path_error_does_not_contain_secret(self) -> None:
        token = make_token({"alg": "RS256", "typ": "JWT"}, {"a": 1}, b"")
        try:
            jwt.decode(token, self.marker, algorithms=["RS256"])
        except PyJWTError as exc:
            self.assertNoMarker(exc)
        else:  # pragma: no cover
            self.fail("decode unexpectedly succeeded")

    def test_hmac_pem_rejection_does_not_contain_key_material(self) -> None:
        pemish = (
            b"-----BEGIN PUBLIC KEY-----\n"
            + self.marker
            + b"\n-----END PUBLIC KEY-----"
        )
        with self.assertRaises(InvalidKeyError) as ctx:
            jwt.encode({"a": 1}, pemish, algorithm="HS256")
        self.assertNoMarker(ctx.exception)

    def test_bad_signature_error_message_is_generic(self) -> None:
        token = jwt.encode({"a": 1}, self.hmac_secret, algorithm="HS256")
        with self.assertRaises(InvalidSignatureError) as ctx:
            jwt.decode(token, self.marker, algorithms=["HS256"])
        self.assertNoMarker(ctx.exception)

    def test_malformed_token_errors_do_not_echo_input(self) -> None:
        for bad in ("not-a-jwt", "a.b", "!!!.!!!.!!!", ""):
            with self.subTest(bad=bad):
                try:
                    jwt.decode(bad, self.marker, algorithms=["HS256"])
                except InvalidTokenError as exc:
                    self.assertNoMarker(exc)
                else:  # pragma: no cover
                    self.fail("decode unexpectedly succeeded")

    def test_duplicate_key_error_does_not_dump_header(self) -> None:
        h = base64url_encode(
            b'{"alg":"HS256","alg":"none","secret":"' + self.marker + b'"}'
        )
        token = b".".join([h, base64url_encode(b"{}"), base64url_encode(b"")]).decode()
        with self.assertRaises(DecodeError) as ctx:
            jwt.get_unverified_header(token)
        self.assertNoMarker(ctx.exception)


class ConcurrentValidationTests(InMemoryKeys, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.setUpKeys()
        cls.verifier = FixedClockPyJWT()
        cls.valid_token = jwt.encode(
            {"exp": FIXED_NOW + 3600}, cls.hmac_secret, algorithm="HS256"
        )
        cls.expired_token = jwt.encode(
            {"exp": FIXED_NOW - 3600}, cls.hmac_secret, algorithm="HS256"
        )
        cls.rsa_token = jwt.encode({"a": 1}, cls.rsa_private, algorithm="RS256")

    def test_concurrent_decode_is_stable_under_threads(self) -> None:
        iterations = 200
        errors: list[BaseException] = []
        results: list[str] = []
        lock = threading.Lock()

        def worker(worker_id: int) -> None:
            for i in range(iterations):
                try:
                    if i % 3 == 0:
                        payload = self.verifier.decode(
                            self.valid_token,
                            self.hmac_secret,
                            algorithms=["HS256"],
                        )
                        outcome = "valid"
                        self.assertIn("exp", payload)
                    elif i % 3 == 1:
                        try:
                            self.verifier.decode(
                                self.expired_token,
                                self.hmac_secret,
                                algorithms=["HS256"],
                            )
                            outcome = "UNEXPECTED-VALID"
                        except ExpiredSignatureError:
                            outcome = "expired"
                    else:
                        payload = self.verifier.decode(
                            self.rsa_token,
                            self.rsa_public_pem,
                            algorithms=["RS256"],
                        )
                        outcome = "rsa"
                        self.assertEqual(payload["a"], 1)
                except BaseException as exc:  # noqa: BLE001
                    with lock:
                        errors.append(exc)
                    return
                with lock:
                    results.append(outcome)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(results), 8 * iterations)
        self.assertNotIn("UNEXPECTED-VALID", results)
        # Every class of outcome must have been observed.
        self.assertEqual(set(results), {"valid", "expired", "rsa"})

    def test_concurrent_algorithm_confusion_attempts_all_fail(self) -> None:
        header = b64url(b'{"alg":"HS256","typ":"JWT"}').encode()
        payload = b64url(b'{"a":1}').encode()
        forged = b".".join(
            [
                header,
                payload,
                base64url_encode(
                    hmac_sign(header + b"." + payload, self.rsa_public_pem)
                ),
            ]
        ).decode()

        failures: list[bool] = []
        barrier = threading.Barrier(8)

        def worker() -> None:
            barrier.wait()
            for _ in range(50):
                try:
                    jwt.decode(forged, self.rsa_public_pem, algorithms=["HS256"])
                    failures.append(False)
                except InvalidKeyError:
                    failures.append(True)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(len(failures), 8 * 50)
        self.assertTrue(all(failures))


if __name__ == "__main__":
    unittest.main(verbosity=2)
