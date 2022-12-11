package com.everyson.mongri_backend.domain.chat.service;

import com.fasterxml.jackson.databind.util.JSONPObject;
import org.json.JSONObject;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.net.URI;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

@Service
public class ChatbotService {
    private String host_url = "http://211.202.222.49:51819";

    public Map generateChatbotRes(Object message) {
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(new MediaType("application", "json"));

        Integer[] speaker = {0, 1};
        JSONObject params = new JSONObject();
        params.put("speaker", speaker);
        params.put("text", message);

        RestTemplate restTemplate = new RestTemplate();

        URI uri = UriComponentsBuilder
                .fromUriString(host_url)
                .path("/chatbot/generate")
                .encode()
                .build()
                .toUri();

        HttpEntity entity = new HttpEntity(params.toString(), headers);

        ResponseEntity<Map> response = restTemplate.postForEntity(uri, entity, Map.class);

        return response.getBody();
    }
}
