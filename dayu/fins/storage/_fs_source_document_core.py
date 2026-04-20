"""文件系统仓储 — 源文档操作 mixin。"""

from __future__ import annotations

from typing import Any, Optional

from dayu.engine.processors.source import Source
from dayu.fins.domain.document_models import (
    DocumentHandle,
    DocumentMeta,
    DocumentQuery,
    DocumentSummary,
    FileObjectMeta,
    FilingCreateRequest,
    FilingDeleteRequest,
    FilingManifestItem,
    FilingRestoreRequest,
    FilingUpdateRequest,
    MaterialCreateRequest,
    MaterialDeleteRequest,
    MaterialManifestItem,
    MaterialRestoreRequest,
    MaterialUpdateRequest,
    SourceDocumentUpsertRequest,
    SourceHandle,
    now_iso8601,
)
from dayu.fins.domain.enums import SourceKind
from dayu.fins.xbrl_file_discovery import has_xbrl_instance

from .local_file_source import LocalFileSource
from ._fs_storage_infra import _FsStorageInfra
from ._fs_storage_utils import (
    _SOURCE_META_FILENAME,
    _build_file_payloads,
    _extract_file_names,
    _extract_file_payloads,
    _file_object_meta_from_dict,
    _guess_media_type,
    _infer_filename_from_uri,
    _list_directory_names,
    _local_path_from_uri,
    _normalize_file_entries,
    _normalize_source_kind,
    _normalize_ticker,
    _read_json_object,
    _resolve_primary_uri,
    _write_json,
)


class _FsSourceDocumentMixin(_FsStorageInfra):
    """源文档（filing / material）操作 mixin。"""

    # ========== material CRUD ==========

    def create_material(self, req: MaterialCreateRequest) -> DocumentHandle:
        """创建材料文档。

        Args:
            req: 材料创建请求。

        Returns:
            文档句柄。

        Raises:
            FileExistsError: 文档已存在时抛出。
            FileNotFoundError: 输入文件不存在时抛出。
            OSError: 写入失败时抛出。
        """

        return self._execute_with_auto_batch(
            req.ticker,
            self._upsert_source_document,
            req,
            SourceKind.MATERIAL,
            True,
        )

    def update_material(self, req: MaterialUpdateRequest) -> DocumentHandle:
        """更新材料文档。

        Args:
            req: 材料更新请求。

        Returns:
            文档句柄。

        Raises:
            FileNotFoundError: 文档或输入文件不存在时抛出。
            OSError: 更新失败时抛出。
        """

        return self._execute_with_auto_batch(
            req.ticker,
            self._upsert_source_document,
            req,
            SourceKind.MATERIAL,
            False,
        )

    def delete_material(self, req: MaterialDeleteRequest) -> None:
        """逻辑删除材料文档。

        Args:
            req: 材料删除请求。

        Returns:
            无。

        Raises:
            FileNotFoundError: 文档不存在时抛出。
            OSError: 写入失败时抛出。
        """

        self._execute_with_auto_batch(
            req.ticker,
            self._toggle_source_deleted,
            req.ticker,
            req.document_id,
            SourceKind.MATERIAL,
            True,
        )

    def restore_material(self, req: MaterialRestoreRequest) -> DocumentHandle:
        """恢复材料文档。

        Args:
            req: 材料恢复请求。

        Returns:
            文档句柄。

        Raises:
            FileNotFoundError: 文档不存在时抛出。
            OSError: 写入失败时抛出。
        """

        return self._execute_with_auto_batch(
            req.ticker,
            self._toggle_source_deleted,
            req.ticker,
            req.document_id,
            SourceKind.MATERIAL,
            False,
        )

    # ========== filing CRUD ==========

    def create_filing(self, req: FilingCreateRequest) -> DocumentHandle:
        """创建财报文档。

        Args:
            req: 财报创建请求。

        Returns:
            文档句柄。

        Raises:
            FileExistsError: 文档已存在时抛出。
            FileNotFoundError: 输入文件不存在时抛出。
            OSError: 写入失败时抛出。
        """

        return self._execute_with_auto_batch(
            req.ticker,
            self._upsert_source_document,
            req,
            SourceKind.FILING,
            True,
        )

    def update_filing(self, req: FilingUpdateRequest) -> DocumentHandle:
        """更新财报文档。

        Args:
            req: 财报更新请求。

        Returns:
            文档句柄。

        Raises:
            FileNotFoundError: 文档或输入文件不存在时抛出。
            OSError: 更新失败时抛出。
        """

        return self._execute_with_auto_batch(
            req.ticker,
            self._upsert_source_document,
            req,
            SourceKind.FILING,
            False,
        )

    def delete_filing(self, req: FilingDeleteRequest) -> None:
        """逻辑删除财报文档。

        Args:
            req: 财报删除请求。

        Returns:
            无。

        Raises:
            FileNotFoundError: 文档不存在时抛出。
            OSError: 写入失败时抛出。
        """

        self._execute_with_auto_batch(
            req.ticker,
            self._toggle_source_deleted,
            req.ticker,
            req.document_id,
            SourceKind.FILING,
            True,
        )

    def restore_filing(self, req: FilingRestoreRequest) -> DocumentHandle:
        """恢复财报文档。

        Args:
            req: 财报恢复请求。

        Returns:
            文档句柄。

        Raises:
            FileNotFoundError: 文档不存在时抛出。
            OSError: 写入失败时抛出。
        """

        return self._execute_with_auto_batch(
            req.ticker,
            self._toggle_source_deleted,
            req.ticker,
            req.document_id,
            SourceKind.FILING,
            False,
        )

    # ========== 查询 ==========

    def get_document_meta(self, ticker: str, document_id: str) -> DocumentMeta:
        """读取文档元数据。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。

        Returns:
            文档元数据字典。

        Raises:
            FileNotFoundError: 元数据不存在时抛出。
            ValueError: 元数据文件内容非法时抛出。
        """

        normalized_ticker = _normalize_ticker(ticker)
        meta_candidates = [
            self._source_meta_path_for_read(normalized_ticker, document_id, SourceKind.FILING),
            self._source_meta_path_for_read(normalized_ticker, document_id, SourceKind.MATERIAL),
            self._processed_meta_path_for_read(normalized_ticker, document_id),
        ]
        for meta_path in meta_candidates:
            if meta_path.exists():
                return _read_json_object(meta_path)
        raise FileNotFoundError(f"document_id={document_id} 的 meta.json 不存在")

    def get_source_meta(self, ticker: str, document_id: str, source_kind: SourceKind) -> DocumentMeta:
        """读取指定来源目录下的源文档元数据。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            source_kind: 来源类型。

        Returns:
            源文档元数据字典。

        Raises:
            FileNotFoundError: 对应来源目录下的 meta.json 不存在时抛出。
            ValueError: 元数据文件内容非法时抛出。
        """

        normalized_ticker = _normalize_ticker(ticker)
        normalized_source_kind = _normalize_source_kind(source_kind)
        meta_path = self._source_meta_path_for_read(normalized_ticker, document_id, normalized_source_kind)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"document_id={document_id} 的 {normalized_source_kind.value} meta.json 不存在"
            )
        return _read_json_object(meta_path)

    def replace_source_meta(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
        meta: DocumentMeta,
    ) -> None:
        """以精确覆盖方式写回源文档元数据。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            source_kind: 来源类型。
            meta: 完整元数据字典。

        Returns:
            无。

        Raises:
            FileNotFoundError: 目标源文档不存在时抛出。
            OSError: 写入失败时抛出。
        """

        normalized_ticker = _normalize_ticker(ticker)
        normalized_source_kind = _normalize_source_kind(source_kind)
        meta_path = self._source_meta_path(normalized_ticker, document_id, normalized_source_kind)
        if not meta_path.exists():
            raise FileNotFoundError(
                f"document_id={document_id} 的 {normalized_source_kind.value} meta.json 不存在"
            )
        normalized_meta = dict(meta)
        _write_json(meta_path, normalized_meta)

        if normalized_source_kind == SourceKind.FILING:
            self.upsert_filing_manifest(
                normalized_ticker,
                [
                    FilingManifestItem(
                        document_id=document_id,
                        internal_document_id=str(normalized_meta.get("internal_document_id", "")),
                        form_type=normalized_meta.get("form_type"),
                        fiscal_year=normalized_meta.get("fiscal_year"),
                        fiscal_period=normalized_meta.get("fiscal_period"),
                        report_date=normalized_meta.get("report_date"),
                        filing_date=normalized_meta.get("filing_date"),
                        amended=bool(normalized_meta.get("amended", False)),
                        ingest_method=str(normalized_meta.get("ingest_method", "upload")),
                        ingest_complete=bool(normalized_meta.get("ingest_complete", True)),
                        is_deleted=bool(normalized_meta.get("is_deleted", False)),
                        deleted_at=normalized_meta.get("deleted_at"),
                        document_version=str(normalized_meta.get("document_version", "v1")),
                        source_fingerprint=str(normalized_meta.get("source_fingerprint", "")),
                        has_xbrl=normalized_meta.get("has_xbrl"),
                    )
                ],
            )
        else:
            self.upsert_material_manifest(
                normalized_ticker,
                [
                    MaterialManifestItem(
                        document_id=document_id,
                        internal_document_id=str(normalized_meta.get("internal_document_id", "")),
                        form_type=normalized_meta.get("form_type"),
                        material_name=normalized_meta.get("material_name"),
                        filing_date=normalized_meta.get("filing_date"),
                        report_date=normalized_meta.get("report_date"),
                        ingest_complete=bool(normalized_meta.get("ingest_complete", True)),
                        is_deleted=bool(normalized_meta.get("is_deleted", False)),
                        deleted_at=normalized_meta.get("deleted_at"),
                        document_version=str(normalized_meta.get("document_version", "v1")),
                        source_fingerprint=str(normalized_meta.get("source_fingerprint", "")),
                    )
                ],
            )

    def list_documents(self, ticker: str, query: DocumentQuery) -> list[DocumentSummary]:
        """从 processed manifest 查询文档摘要。

        Args:
            ticker: 股票代码。
            query: 查询条件。

        Returns:
            文档摘要列表。

        Raises:
            OSError: 读取 manifest 失败时抛出。
            ValueError: manifest 内容非法时抛出。
        """

        normalized_ticker = _normalize_ticker(ticker)
        manifest = self._read_manifest(self._processed_manifest_path_for_read(normalized_ticker), normalized_ticker)
        result: list[DocumentSummary] = []
        for item in manifest["documents"]:
            summary = DocumentSummary.from_dict(item)
            if not query.include_deleted and summary.is_deleted:
                continue
            if query.source_kind and summary.source_kind != query.source_kind:
                continue
            if query.form_type and summary.form_type != query.form_type:
                continue
            if query.fiscal_years and summary.fiscal_year not in query.fiscal_years:
                continue
            if query.fiscal_periods and summary.fiscal_period not in query.fiscal_periods:
                continue
            result.append(summary)
        return result

    def list_document_ids(self, ticker: str, source_kind: Optional[SourceKind] = None) -> list[str]:
        """列出文档 ID。

        Args:
            ticker: 股票代码。
            source_kind: 可选来源类型过滤。

        Returns:
            已排序文档 ID 列表。

        Raises:
            OSError: 读取目录失败时抛出。
        """

        normalized_ticker = _normalize_ticker(ticker)
        if source_kind == SourceKind.FILING:
            return _list_directory_names(self._source_root_for_read(normalized_ticker, SourceKind.FILING))
        if source_kind == SourceKind.MATERIAL:
            return _list_directory_names(self._source_root_for_read(normalized_ticker, SourceKind.MATERIAL))

        filings = _list_directory_names(self._source_root_for_read(normalized_ticker, SourceKind.FILING))
        materials = _list_directory_names(self._source_root_for_read(normalized_ticker, SourceKind.MATERIAL))
        return sorted(set(filings + materials))

    def has_source_storage_root(self, ticker: str, source_kind: SourceKind) -> bool:
        """判断某类源文档根目录是否存在且为目录。

        Args:
            ticker: 股票代码。
            source_kind: 来源类型。

        Returns:
            若目录存在且为目录则返回 `True`，不存在返回 `False`。

        Raises:
            NotADirectoryError: 根路径存在但不是目录时抛出。
            OSError: 文件系统访问失败时抛出。
        """

        normalized_ticker = _normalize_ticker(ticker)
        normalized_source_kind = _normalize_source_kind(source_kind)
        root = self._source_root_for_read(normalized_ticker, normalized_source_kind)
        if not root.exists():
            return False
        if not root.is_dir():
            raise NotADirectoryError(f"source root 不是目录: {root}")
        return True

    def has_filing_xbrl_instance(self, ticker: str, document_id: str) -> bool:
        """判断 filing 目录下是否已落盘 XBRL instance 文件。

        Args:
            ticker: 股票代码。
            document_id: filing 文档 ID。

        Returns:
            若存在 XBRL instance 文件则返回 `True`，否则返回 `False`。

        Raises:
            FileNotFoundError: filing 目录不存在时抛出。
            NotADirectoryError: filing 路径存在但不是目录时抛出。
            OSError: 文件系统访问失败时抛出。
        """

        normalized_ticker = _normalize_ticker(ticker)
        filing_dir = self._source_root_for_read(normalized_ticker, SourceKind.FILING) / document_id
        if not filing_dir.exists():
            raise FileNotFoundError(f"filing 目录不存在: {filing_dir}")
        if not filing_dir.is_dir():
            raise NotADirectoryError(f"filing 路径不是目录: {filing_dir}")
        return has_xbrl_instance(filing_dir)

    # ========== handle & 文件访问 ==========

    def get_source_handle(self, ticker: str, document_id: str, source_kind: SourceKind) -> SourceHandle:
        """获取源文档句柄。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            source_kind: 来源类型。

        Returns:
            源文档句柄。

        Raises:
            FileNotFoundError: 文档不存在时抛出。
        """

        normalized_ticker = _normalize_ticker(ticker)
        normalized_source_kind = _normalize_source_kind(source_kind)
        meta_path = self._source_meta_path_for_read(normalized_ticker, document_id, normalized_source_kind)
        if not meta_path.exists():
            raise FileNotFoundError(f"document_id={document_id} 不存在于 {normalized_source_kind}")
        return SourceHandle(
            ticker=normalized_ticker,
            document_id=document_id,
            source_kind=normalized_source_kind.value,
        )

    def get_primary_file(self, handle: SourceHandle) -> FileObjectMeta:
        """获取源文档主文件元数据。

        Args:
            handle: 源文档句柄。

        Returns:
            主文件元数据。

        Raises:
            FileNotFoundError: 主文件无法定位时抛出。
            ValueError: 元数据格式非法时抛出。
        """

        meta = self._get_handle_meta(handle)
        files = meta.get("files", [])
        if not isinstance(files, list):
            raise ValueError("meta.files 必须为 list")
        if not files:
            raise FileNotFoundError("源文档未绑定文件，无法定位主文件")
        primary_name = str(meta.get("primary_document", "")).strip()
        if primary_name:
            for item in files:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or _infer_filename_from_uri(item.get("uri", ""))).strip()
                if name == primary_name:
                    return _file_object_meta_from_dict(item)
        for item in files:
            if isinstance(item, dict):
                return _file_object_meta_from_dict(item)
        raise FileNotFoundError("源文档未找到可用文件条目")

    def get_source(self, handle: SourceHandle, file_meta: FileObjectMeta) -> Source:
        """根据文件元数据获取 Source。

        Args:
            handle: 源文档句柄。
            file_meta: 文件元数据。

        Returns:
            Source 抽象。

        Raises:
            ValueError: 文件元数据非法时抛出。
            OSError: 构建 Source 失败时抛出。
        """

        uri = str(file_meta.uri or "").strip()
        if not uri:
            raise ValueError("file_meta.uri 不能为空")
        path = _local_path_from_uri(self.portfolio_root, uri)
        media_type = file_meta.content_type or _guess_media_type(path)
        return LocalFileSource(
            path=path,
            uri=uri,
            media_type=media_type,
            content_length=file_meta.size,
            etag=file_meta.etag,
        )

    def get_primary_source(self, ticker: str, document_id: str, source_kind: SourceKind) -> Source:
        """获取源文档主文件的 Source。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            source_kind: 来源类型。

        Returns:
            Source 抽象。

        Raises:
            FileNotFoundError: 文档或主文件不存在时抛出。
            ValueError: 文件元数据非法时抛出。
            OSError: 构建 Source 失败时抛出。
        """

        handle = self.get_source_handle(ticker=ticker, document_id=document_id, source_kind=source_kind)
        primary_file = self.get_primary_file(handle)
        return self.get_source(handle, primary_file)

    # ========== 内部实现 ==========

    def _upsert_source_document(
        self,
        req: SourceDocumentUpsertRequest,
        source_kind: SourceKind,
        is_create: bool,
    ) -> DocumentHandle:
        """创建或更新源文档。

        Args:
            req: 源文档写入请求。
            source_kind: 文档来源类型。
            is_create: 是否创建流程。

        Returns:
            文档句柄。

        Raises:
            FileExistsError: 创建时文档已存在。
            FileNotFoundError: 更新时文档不存在或拷贝文件不存在。
            OSError: 写入失败。
        """

        ticker = _normalize_ticker(req.ticker)
        source_root = self._source_root(ticker, source_kind)
        source_root.mkdir(parents=True, exist_ok=True)
        document_dir = source_root / req.document_id
        meta_path = document_dir / _SOURCE_META_FILENAME

        meta_exists = meta_path.exists()
        if is_create and meta_exists:
            raise FileExistsError(f"文档已存在: {meta_path}")
        if not is_create and not meta_exists:
            raise FileNotFoundError(f"文档不存在: {meta_path}")

        document_dir.mkdir(parents=True, exist_ok=True)
        previous_meta = _read_json_object(meta_path) if meta_path.exists() else {}

        previous_files = _extract_file_payloads(previous_meta)
        if req.file_entries is not None:
            file_payloads = _normalize_file_entries(req.file_entries)
        elif req.files:
            file_payloads = _build_file_payloads(req.files)
        else:
            file_payloads = previous_files
        now = now_iso8601()

        merged_meta = dict(previous_meta)
        merged_meta.update(req.meta)
        merged_meta["ticker"] = ticker
        merged_meta["document_id"] = req.document_id
        merged_meta["internal_document_id"] = req.internal_document_id
        merged_meta["form_type"] = req.form_type or merged_meta.get("form_type")
        merged_meta["updated_at"] = now
        merged_meta.setdefault("created_at", now)
        merged_meta.setdefault("first_ingested_at", now)
        merged_meta.setdefault("ingest_complete", True)
        merged_meta.setdefault("is_deleted", False)
        merged_meta.setdefault("deleted_at", None)
        merged_meta.setdefault("document_version", "v1")
        merged_meta.setdefault("source_fingerprint", "")

        selected_primary_document = self._select_primary_document(
            explicit_primary=req.primary_document,
            previous_primary=previous_meta.get("primary_document"),
            current_file_names=_extract_file_names(file_payloads),
            previous_file_names=_extract_file_names(previous_files),
        )
        if selected_primary_document is not None:
            merged_meta["primary_document"] = selected_primary_document
        merged_meta["files"] = file_payloads

        _write_json(meta_path, merged_meta)

        if source_kind == SourceKind.FILING:
            self.upsert_filing_manifest(
                ticker,
                [
                    FilingManifestItem(
                        document_id=req.document_id,
                        internal_document_id=req.internal_document_id,
                        form_type=merged_meta.get("form_type"),
                        fiscal_year=merged_meta.get("fiscal_year"),
                        fiscal_period=merged_meta.get("fiscal_period"),
                        report_date=merged_meta.get("report_date"),
                        filing_date=merged_meta.get("filing_date"),
                        amended=bool(merged_meta.get("amended", False)),
                        ingest_method=str(merged_meta.get("ingest_method", "upload")),
                        ingest_complete=bool(merged_meta.get("ingest_complete", True)),
                        is_deleted=bool(merged_meta.get("is_deleted", False)),
                        deleted_at=merged_meta.get("deleted_at"),
                        document_version=str(merged_meta.get("document_version", "v1")),
                        source_fingerprint=str(merged_meta.get("source_fingerprint", "")),
                        has_xbrl=merged_meta.get("has_xbrl"),
                    )
                ],
            )
        else:
            self.upsert_material_manifest(
                ticker,
                [
                    MaterialManifestItem(
                        document_id=req.document_id,
                        internal_document_id=req.internal_document_id,
                        form_type=merged_meta.get("form_type"),
                        material_name=merged_meta.get("material_name"),
                        filing_date=merged_meta.get("filing_date"),
                        report_date=merged_meta.get("report_date"),
                        ingest_complete=bool(merged_meta.get("ingest_complete", True)),
                        is_deleted=bool(merged_meta.get("is_deleted", False)),
                        deleted_at=merged_meta.get("deleted_at"),
                        document_version=str(merged_meta.get("document_version", "v1")),
                        source_fingerprint=str(merged_meta.get("source_fingerprint", "")),
                    )
                ],
            )

        primary_file_uri = _resolve_primary_uri(file_payloads, selected_primary_document)
        return DocumentHandle(
            ticker=ticker,
            document_id=req.document_id,
            form_type=merged_meta.get("form_type"),
            primary_file_uri=primary_file_uri,
            file_uris=[str(item.get("uri")) for item in file_payloads if isinstance(item, dict)],
        )

    def _toggle_source_deleted(
        self,
        ticker: str,
        document_id: str,
        source_kind: SourceKind,
        deleted: bool,
    ) -> DocumentHandle:
        """切换源文档逻辑删除状态。

        Args:
            ticker: 股票代码。
            document_id: 文档 ID。
            source_kind: 来源类型。
            deleted: 目标删除状态。

        Returns:
            更新后的文档句柄。

        Raises:
            FileNotFoundError: 文档不存在。
            OSError: 写入失败。
        """

        normalized_ticker = _normalize_ticker(ticker)
        meta_path = self._source_meta_path(normalized_ticker, document_id, source_kind)
        if not meta_path.exists():
            raise FileNotFoundError(f"文档不存在: {meta_path}")

        meta = _read_json_object(meta_path)
        meta["is_deleted"] = deleted
        meta["deleted_at"] = now_iso8601() if deleted else None
        meta["updated_at"] = now_iso8601()
        _write_json(meta_path, meta)

        if source_kind == SourceKind.FILING:
            self.upsert_filing_manifest(
                normalized_ticker,
                [
                    FilingManifestItem(
                        document_id=document_id,
                        internal_document_id=str(meta.get("internal_document_id", "")),
                        form_type=meta.get("form_type"),
                        fiscal_year=meta.get("fiscal_year"),
                        fiscal_period=meta.get("fiscal_period"),
                        report_date=meta.get("report_date"),
                        filing_date=meta.get("filing_date"),
                        amended=bool(meta.get("amended", False)),
                        ingest_method=str(meta.get("ingest_method", "upload")),
                        ingest_complete=bool(meta.get("ingest_complete", True)),
                        is_deleted=bool(meta.get("is_deleted", False)),
                        deleted_at=meta.get("deleted_at"),
                        document_version=str(meta.get("document_version", "v1")),
                        source_fingerprint=str(meta.get("source_fingerprint", "")),
                        has_xbrl=meta.get("has_xbrl"),
                    )
                ],
            )
        else:
            self.upsert_material_manifest(
                normalized_ticker,
                [
                    MaterialManifestItem(
                        document_id=document_id,
                        internal_document_id=str(meta.get("internal_document_id", "")),
                        form_type=meta.get("form_type"),
                        material_name=meta.get("material_name"),
                        filing_date=meta.get("filing_date"),
                        report_date=meta.get("report_date"),
                        ingest_complete=bool(meta.get("ingest_complete", True)),
                        is_deleted=bool(meta.get("is_deleted", False)),
                        deleted_at=meta.get("deleted_at"),
                        document_version=str(meta.get("document_version", "v1")),
                        source_fingerprint=str(meta.get("source_fingerprint", "")),
                    )
                ],
            )

        file_payloads = _extract_file_payloads(meta)
        return DocumentHandle(
            ticker=normalized_ticker,
            document_id=document_id,
            form_type=meta.get("form_type"),
            primary_file_uri=_resolve_primary_uri(
                file_payloads,
                str(meta.get("primary_document", "")).strip() or None,
            ),
            file_uris=[str(item.get("uri")) for item in file_payloads if isinstance(item, dict)],
        )
