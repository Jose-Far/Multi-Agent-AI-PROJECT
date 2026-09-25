import logging
from urllib.parse import urlparse
from typing import Dict, Any, List
import dns.resolver
import dns.exception

logger = logging.getLogger(__name__)

class DNSCollector:
    """
    Production-grade DNS Collector module for gathering multi-vector DNS records
    (A, AAAA, MX, TXT, NS, CNAME, SOA) and TTL statistics to support DNS AI Agent
    and campaign-level infrastructure correlation.
    """
    def __init__(self, timeout: float = 4.0):
        self.timeout = timeout
        self.resolver = dns.resolver.Resolver()
        self.resolver.timeout = self.timeout
        self.resolver.lifetime = self.timeout
        # Public secure resolvers fallback list
        self.resolver.nameservers = ['1.1.1.1', '8.8.8.8', '9.9.9.9']

    def collect(self, target_url: str) -> Dict[str, Any]:
        """
        Executes deep DNS enumeration across multiple record types for the target domain.
        
        Args:
            target_url (str): The URL or hostname to inspect.
            
        Dict[str, Any]: Standardized telemetry payload containing status, error flags, 
                        and rich DNS structural metrics.
        """
        hostname = urlparse(target_url).hostname or target_url
        if not hostname:
            return self._build_error_payload("Invalid hostname provided for DNS resolution.")

        # Normalize domain by stripping leading 'www.' for accurate zone/root lookups
        domain = hostname.replace("www.", "") if hostname.startswith("www.") else hostname

        dns_records: Dict[str, List[Any]] = {
            "A": [],
            "AAAA": [],
            "MX": [],
            "TXT": [],
            "NS": [],
            "CNAME": [],
            "SOA": []
        }
        ttl_metrics: Dict[str, int] = {}
        resolution_success = False

        # 1. Gather A Records (IPv4 Addresses)
        try:
            answers = self.resolver.resolve(domain, 'A')
            dns_records["A"] = [rdata.address for rdata in answers]
            if answers.rrset:
                ttl_metrics["A_TTL"] = answers.rrset.ttl
            resolution_success = True
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            logger.debug(f"No A records found for domain: {domain}")
        except Exception as e:
            logger.warning(f"Error querying A records for {domain}: {str(e)}")

        # 2. Gather AAAA Records (IPv6 Addresses)
        try:
            answers = self.resolver.resolve(domain, 'AAAA')
            dns_records["AAAA"] = [rdata.address for rdata in answers]
            if answers.rrset:
                ttl_metrics["AAAA_TTL"] = answers.rrset.ttl
        except Exception:
            pass

        # 3. Gather MX Records (Mail Exchangers & Priority)
        try:
            answers = self.resolver.resolve(domain, 'MX')
            dns_records["MX"] = [
                {"preference": rdata.preference, "exchange": str(rdata.exchange).rstrip('.')}
                for rdata in answers
            ]
            if answers.rrset:
                ttl_metrics["MX_TTL"] = answers.rrset.ttl
        except Exception:
            pass

        # 4. Gather TXT Records (SPF, DMARC, Domain Ownership Proofs)
        try:
            answers = self.resolver.resolve(domain, 'TXT')
            dns_records["TXT"] = [str(rdata).strip('"') for rdata in answers]
            if answers.rrset:
                ttl_metrics["TXT_TTL"] = answers.rrset.ttl
        except Exception:
            pass

        # 5. Gather NS Records (Authoritative Name Servers)
        try:
            answers = self.resolver.resolve(domain, 'NS')
            dns_records["NS"] = [str(rdata.target).rstrip('.') for rdata in answers]
            if answers.rrset:
                ttl_metrics["NS_TTL"] = answers.rrset.ttl
        except Exception:
            pass

        # 6. Gather CNAME Records (Canonical Name Aliases)
        try:
            answers = self.resolver.resolve(domain, 'CNAME')
            dns_records["CNAME"] = [str(rdata.target).rstrip('.') for rdata in answers]
            if answers.rrset:
                ttl_metrics["CNAME_TTL"] = answers.rrset.ttl
        except Exception:
            pass

        # 7. Gather SOA Records (Start of Authority)
        try:
            answers = self.resolver.resolve(domain, 'SOA')
            dns_records["SOA"] = [
                {
                    "mname": str(rdata.mname).rstrip('.'),
                    "rname": str(rdata.rname).rstrip('.'),
                    "serial": rdata.serial,
                    "refresh": rdata.refresh,
                    "retry": rdata.retry,
                    "expire": rdata.expire,
                    "minimum": rdata.minimum
                }
                for rdata in answers
            ]
        except Exception:
            pass

        if not resolution_success and not dns_records["CNAME"]:
            return self._build_error_payload(f"Domain '{domain}' failed to resolve basic routing records.")

        return {
            "success": True,
            "error": None,
            "data": {
                "queried_domain": domain,
                "records": dns_records,
                "ttl_statistics": ttl_metrics,
                "record_counts": {
                    "a_count": len(dns_records["A"]),
                    "mx_count": len(dns_records["MX"]),
                    "ns_count": len(dns_records["NS"]),
                    "txt_count": len(dns_records["TXT"])
                }
            }
        }

    def _build_error_payload(self, message: str) -> Dict[str, Any]:
        """Constructs standardized error response structure."""
        logger.error(f"DNSCollector Error: {message}")
        return {
            "success": False,
            "error": message,
            "data": {
                "queried_domain": "",
                "records": {},
                "ttl_statistics": {},
                "record_counts": {}
            }
        }