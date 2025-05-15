import requests
from grafana_api.grafana_face import GrafanaFace
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from prometheus_client.exposition import basic_auth_handler
from requests.auth import HTTPBasicAuth

GRAFANA_URL = "149.28.73.86:9091"
PROMETHUS_URL = "149.28.73.86:9001"
GRAFANA_API = "149.28.73.86:3000"
GRAFANA_USERNAME = "admin"
GRAFANA_PASSWORD = "admin"


class Promethus_Grafana:
    def __init__(self, host_grafana, url_grafana, url_promethus):
        self.username = GRAFANA_USERNAME
        self.password = GRAFANA_PASSWORD
        self.url_grafana = url_grafana
        self.url_promethus = url_promethus
        self.host_grafana = host_grafana

        self.grafana_api = GrafanaFace(
            auth=(self.username, self.password), host=url_grafana
        )

    def promethus_push_to(self, job_name):
        def my_auth_handler(url, method, timeout, headers, data):
            return basic_auth_handler(
                url, method, timeout, headers, data, self.username, self.password
            )

        registry = CollectorRegistry()
        g = Gauge(
            "job_last_success_unixtime",
            "Last time a batch job successfully finished",
            registry=registry,
        )
        g.set_to_current_time()
        push_to_gateway(
            GRAFANA_URL, job=job_name, registry=registry, handler=my_auth_handler
        )

    def create_dashboard(self, name_dashboard, title, job_query, tag):
        payload = {
            "dashboard": {
                "annotations": {
                    "list": [
                        {
                            "builtIn": 1,
                            "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                            "enable": True,
                            "hide": True,
                            "iconColor": "rgba(0, 211, 255, 1)",
                            "name": "Annotations & Alerts",
                            "type": "dashboard",
                        }
                    ]
                },
                "editable": True,
                "fiscalYearStartMonth": 0,
                "graphTooltip": 0,
                "links": [],
                "panels": [
                    {
                        "datasource": {
                            "type": "prometheus",
                            "uid": "PBFA97CFB590B2093",
                        },
                        "fieldConfig": {
                            "defaults": {
                                "color": {"mode": "palette-classic"},
                                "custom": {
                                    "axisBorderShow": False,
                                    "axisCenteredZero": False,
                                    "axisColorMode": "text",
                                    "axisLabel": "",
                                    "axisPlacement": "auto",
                                    "barAlignment": 0,
                                    "drawStyle": "line",
                                    "fillOpacity": 0,
                                    "gradientMode": "none",
                                    "hideFrom": {
                                        "legend": False,
                                        "tooltip": False,
                                        "viz": False,
                                    },
                                    "insertNulls": False,
                                    "lineInterpolation": "linear",
                                    "lineWidth": 1,
                                    "pointSize": 5,
                                    "scaleDistribution": {"type": "linear"},
                                    "showPoints": "auto",
                                    "spanNulls": False,
                                    "stacking": {"group": "A", "mode": "none"},
                                    "thresholdsStyle": {"mode": "off"},
                                },
                                "mappings": [],
                                "thresholds": {
                                    "mode": "absolute",
                                    "steps": [
                                        {"color": "green", "value": None},
                                        {"color": "red", "value": 80},
                                    ],
                                },
                            },
                            "overrides": [],
                        },
                        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
                        "id": 1,
                        "options": {
                            "legend": {
                                "calcs": [],
                                "displayMode": "list",
                                "placement": "bottom",
                                "showLegend": True,
                            },
                            "tooltip": {
                                "maxHeight": 600,
                                "mode": "single",
                                "sort": "none",
                            },
                        },
                        "targets": [
                            {
                                "expr": 'job_last_success_unixtime{job="training"}',  # Ví dụ truy vấn Prometheus
                                "format": "time_series",
                                "interval": "1m",
                                "legendFormat": "{{job}}",
                                "refId": "A",
                            }
                        ],
                        "title": title,
                        "type": "timeseries",
                    }
                ],
                "schemaVersion": 39,
                "tags": tag,
                "templating": {"list": []},
                "time": {"from": "now-6h", "to": "now"},
                "timeRangeUpdatedDuringEditOrView": False,
                "timepicker": {},
                "timezone": "browser",
                "title": name_dashboard,
                "version": 0,
                "weekStart": "",
                "id": None,
                "uid": "",
            },
            "folderUid": "",
            "message": "",
            "overwrite": False,
        }

        url = f"{self.host_grafana}/api/dashboards/db"

        # Headers
        headers = {
            "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
            "accept": "application/json, text/plain, */*",
        }

        # Thực hiện yêu cầu GET với Basic Auth
        res = requests.post(
            url,
            headers=headers,
            json=payload,
            auth=HTTPBasicAuth(GRAFANA_USERNAME, GRAFANA_PASSWORD),
        )
        print(res)

    def generate_link_public(self, tag):
        dashboard_tag = self.grafana_api.search.search_dashboards(tag=tag)
        print(dashboard_tag)
        dashboard_uid = dashboard_tag[0]["uid"]
        print(dashboard_uid)

        # URL API để công khai Dashboard
        url = (
            f"{self.host_grafana}/api/dashboards/uid/{dashboard_uid}/public-dashboards"
        )

        # Headers
        headers = {
            "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
            "accept": "application/json, text/plain, */*",
        }

        # Thực hiện yêu cầu GET với Basic Auth
        requests.post(
            url,
            headers=headers,
            json={"isEnabled": True},
            auth=HTTPBasicAuth("admin", "admin"),
        )
        response = requests.get(
            url, headers=headers, auth=HTTPBasicAuth("admin", "admin")
        )

        if response.status_code == 200:
            res = response.json()
            link_url = f'http://{GRAFANA_API}/public-dashboards/{res["accessToken"]}'
        else:
            return False

        return link_url


promethus_grafana = Promethus_Grafana(
    host_grafana="http://{GRAFANA_API}",
    url_grafana=GRAFANA_API,
    url_promethus=PROMETHUS_URL,
)
# promethus_grafana.promethus_push_to("training")
# # promethus_grafana.create_dashboard("ML_monitor_2", "Training Job", "training00", ["ml_00"])
# link = promethus_grafana.generate_link_public("ml_00")
