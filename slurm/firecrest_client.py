"""FirecREST client wrapper for submitting and monitoring jobs on CSCS clusters."""

import os
from functools import cached_property

from dotenv import load_dotenv
import firecrest as f7t


TOKEN_URI = "https://auth.cscs.ch/auth/realms/firecrest-clients/protocol/openid-connect/token"

PLATFORM_URLS = {
    "ml": "https://api.cscs.ch/ml/firecrest/v2",
    "hpc": "https://api.cscs.ch/hpc/firecrest/v2",
    "cw": "https://api.cscs.ch/cw/firecrest/v2",
}

# Which platform each cluster belongs to
CLUSTER_PLATFORM = {
    "clariden": "ml",
    "bristen": "ml",
    "daint": "hpc",
    "eiger": "hpc",
    "santis": "cw",
}


class FirecrestClient:
    """Thin wrapper around pyFirecREST v2 with env-based auth."""

    def __init__(self):
        load_dotenv()
        self._client_id = os.environ["FIRECREST_CONSUMER"]
        self._client_secret = os.environ["FIRECREST_SECRET"]

    def _get_client(self, platform: str) -> f7t.v2.Firecrest:
        url = PLATFORM_URLS[platform]
        return f7t.v2.Firecrest(
            firecrest_url=url,
            authorization=f7t.ClientCredentialsAuth(
                client_id=self._client_id,
                client_secret=self._client_secret,
                token_uri=TOKEN_URI,
            ),
        )

    def client_for(self, cluster: str) -> f7t.v2.Firecrest:
        """Get a FirecREST client for the given cluster."""
        platform = CLUSTER_PLATFORM[cluster]
        return self._get_client(platform)

    def submit(self, cluster: str, script: str, working_dir: str, account: str) -> dict:
        """Submit a batch script to the given cluster.

        Args:
            cluster: Cluster name (e.g. "clariden").
            script: Full rendered batch script content.
            working_dir: Remote working directory for the job.
            account: Slurm account to charge.

        Returns:
            Dict with at least 'jobId'.
        """
        client = self.client_for(cluster)
        return client.submit(
            system_name=cluster,
            working_dir=working_dir,
            script_str=script,
            account=account,
        )

    def job_info(self, cluster: str, jobid: str | None = None) -> list:
        """Query job info. If jobid is None, returns all user jobs."""
        client = self.client_for(cluster)
        return client.job_info(system_name=cluster, jobid=jobid)

    def cancel(self, cluster: str, jobid: str) -> dict:
        """Cancel a job."""
        client = self.client_for(cluster)
        return client.cancel_job(system_name=cluster, jobid=jobid)

    def list_files(self, cluster: str, path: str) -> list:
        """List files at a remote path."""
        client = self.client_for(cluster)
        return client.list_files(system_name=cluster, path=path)

    def download(self, cluster: str, remote_path: str, local_path: str):
        """Download a file from the cluster."""
        client = self.client_for(cluster)
        return client.download(
            system_name=cluster, source_path=remote_path, target_path=local_path
        )

    def head(self, cluster: str, path: str, num_lines: int = 100) -> str:
        """Read the first N lines of a remote file."""
        client = self.client_for(cluster)
        result = client.head(system_name=cluster, path=path, num_lines=num_lines)
        return result.get("content", str(result))
